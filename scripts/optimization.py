#!/usr/bin/env python3
"""
Optimization module for multi-year DC OPF with annualized (annuity) investments.
No layering: create, solve, extract numeric results, store in integrated_network.
"""

import cvxpy as cp
import pandas as pd
import numpy as np
import os
import logging

# Configure only a file handler, no console handler
logger = logging.getLogger(__name__)
# Don't add any handlers here - we'll configure through the main module

def compute_crf(lifetime, discount_rate):
    """
    Compute the capital recovery factor (CRF) for an asset.
    
    CRF = i * (1+i)^n / ((1+i)^n - 1)
    where:
      i = discount_rate,
      n = lifetime (years)
      
    Returns 1.0 if lifetime is zero or close to zero.
    """
    if lifetime is None or lifetime <= 0:
        return 1.0
    i = discount_rate
    n = lifetime
    numerator = i * (1 + i)**n
    denominator = (1 + i)**n - 1
    if abs(denominator) < 1e-9:
        return 1.0
    return numerator / denominator

def compute_discount_sum(lifetime, discount_rate):
    """
    Compute the sum of discounted factors over the asset's lifetime:
    
    A = sum_{t=1}^{n} 1/(1+i)^t = (1 - 1/(1+i)^n) / i  if i>0
    """
    if discount_rate <= 0:
        return lifetime
    return (1 - 1 / ((1 + discount_rate) ** lifetime)) / discount_rate

def dcopf(integrated_network):
    """
    Create an integrated multi-year DC OPF problem with:
      - Separate 'build' and 'installed' binary variables per asset & year.
      - Generator dispatch & storage variables for each year+season.
      - Summed operational + annualized capital cost objective (cost based on 'build').
      - Lifetimes handled by linking 'build' to 'installed' status.
        
    Returns:
      problem_dict = {
        'objective': cp.Minimize(...),
        'constraints': [...],
        'variables': {
          'gen_build': {...},        # New build variables
          'storage_build': {...},  # New build variables
          'gen_installed': {...},    # Operational status
          'storage_installed': {...},# Operational status
          'season_variables': {season: {...}}
        },
        'years': [...],
        'seasons': [...],
      }
    """
    years = integrated_network.years
    seasons = list(integrated_network.season_networks.keys())
    first_network = list(integrated_network.season_networks.values())[0] if seasons else None

    generators = first_network.generators.index if first_network else []
    storage_units = first_network.storage_units.index if first_network else []
    buses = first_network.buses.index if first_network else []
    lines = first_network.lines.index if first_network else []

    # -- 1) Create global 'build' and 'installed' variables
    gen_build = {(g, y): cp.Variable(boolean=True, name=f"gen_build_{g}_{y}") for g in generators for y in years}
    storage_build = {(s, y): cp.Variable(boolean=True, name=f"storage_build_{s}_{y}") for s in storage_units for y in years}
    
    gen_installed = {(g, y): cp.Variable(boolean=True, name=f"gen_installed_{g}_{y}") for g in generators for y in years}
    storage_installed = {(s, y): cp.Variable(boolean=True, name=f"storage_installed_{s}_{y}") for s in storage_units for y in years}

    # -- 2) Link 'build' to 'installed' status based on lifetime
    global_constraints = []
    
    # For each generator g, link installed[g, y] to build decisions within its lifetime
    for g in generators:
        lifetime_g = first_network.generators.at[g, 'lifetime_years']
        # Ensure lifetime is treated as an integer for range calculation
        lifetime_g_int = int(lifetime_g) if pd.notna(lifetime_g) else 0 
        
        # ---- NEW: "at-most-one-build-per-lifetime window" ----
        for y_idx, y in enumerate(years):
            window_builds = [gen_build[(g, yb)]
                             for yb_idx, yb in enumerate(years)
                             if (y_idx - yb_idx) < lifetime_g_int and y_idx >= yb_idx]
            global_constraints.append(cp.sum(window_builds) <= 1)
        
        # ---- Equality definition of installed ----
        for y_idx, y in enumerate(years):
            window_builds = [gen_build[(g, yb)]
                             for yb_idx, yb in enumerate(years)
                             if (y_idx - yb_idx) < lifetime_g_int and y_idx >= yb_idx]
            global_constraints.append(
                gen_installed[(g, y)] == cp.sum(window_builds))

    # For each storage unit s, link installed[s, y] to build decisions within its lifetime
    for s in storage_units:
        lifetime_s = first_network.storage_units.at[s, 'lifetime_years']
        # Ensure lifetime is treated as an integer
        lifetime_s_int = int(lifetime_s) if pd.notna(lifetime_s) else 0
        
        # ---- NEW: "at-most-one-build-per-lifetime window" ----
        for y_idx, y in enumerate(years):
            window_builds = [storage_build[(s, yb)]
                             for yb_idx, yb in enumerate(years)
                             if (y_idx - yb_idx) < lifetime_s_int and y_idx >= yb_idx]
            global_constraints.append(cp.sum(window_builds) <= 1)
        
        # ---- Equality definition of installed ----
        for y_idx, y in enumerate(years):
            window_builds = [storage_build[(s, yb)]
                             for yb_idx, yb in enumerate(years)
                             if (y_idx - yb_idx) < lifetime_s_int and y_idx >= yb_idx]
            global_constraints.append(
                storage_installed[(s, y)] == cp.sum(window_builds))

    # Add mutual exclusion constraints for technology pairs
    # Check if generators have a 'mutual_exclusion_group' column to define groups
    if first_network and 'mutual_exclusion_group' in first_network.generators.columns:
        # Group generators by their mutual exclusion group
        exclusion_groups = {}
        for g in generators:
            if g in first_network.generators.index:
                group = first_network.generators.at[g, 'mutual_exclusion_group']
                if pd.notna(group) and group != '':
                    if group not in exclusion_groups:
                        exclusion_groups[group] = []
                    exclusion_groups[group].append(g)
        
        # Add constraints for each group
        for group, group_generators in exclusion_groups.items():
            if len(group_generators) > 1:  # Only add constraint if group has multiple generators
                for y in years:
                    global_constraints.append(
                        cp.sum([gen_installed[(g, y)] for g in group_generators]) <= 1
                    )

    # -- 3) Create flat variable dictionaries and constraints
    # ----------------------------------------------------------
    # Flat dictionaries indexed (asset, year, season)
    # ----------------------------------------------------------
    p_gen, p_line = {}, {}
    p_charge, p_discharge, soc = {}, {}, {}

    flat_constraints = []             # replaces season_constraints

    # Build lookup dictionaries to keep loops light
    mc_g = {}  # Marginal costs for generators
    for g in generators:
        if g in first_network.generators.index:
            mc_g[g] = first_network.generators.at[g, 'marginal_cost']

    # Pre-compute load dictionaries and bus connections
    bus_load_dict = {}
    g_at_bus = {}
    s_at_bus = {}
    lines_from = {}
    lines_to = {}

    for season in seasons:
        net = integrated_network.season_networks[season]
        T = net.T
        
        # Initialize load dictionary for this season
        bus_load_dict[season] = {b: np.zeros(T) for b in buses}
        g_at_bus[season] = {b: [] for b in buses}
        s_at_bus[season] = {b: [] for b in buses}
        lines_from[season] = {b: [] for b in buses}
        lines_to[season] = {b: [] for b in buses}
        
        # Build load dictionary
        if not net.loads.empty:
            for ld in net.loads.index:
                load_bus_orig = net.loads.at[ld, 'bus']
                # Try to match bus types (int vs str)
                matched_bus = None
                for b in buses:
                   try:
                       if str(load_bus_orig) == str(b):
                           matched_bus = b
                           break
                   except: # Handle potential type errors during comparison
                        pass
                
                if matched_bus is not None:
                    if 'p' in net.loads_t and ld in net.loads_t['p']:
                        load_ts = net.loads_t['p'][ld].values[:T]
                        bus_load_dict[season][matched_bus] += load_ts
                    else:
                        static_val = net.loads.at[ld, 'p_mw']
                        if pd.notna(static_val):
                            bus_load_dict[season][matched_bus] += np.ones(T) * static_val
                        else:
                            logger.warning(f"Load {ld} at bus {matched_bus} has invalid p_mw value: {static_val}. Setting to 0.")
        
        # Build connection dictionaries
        for g in generators:
            if g in net.generators.index:
                bus_g = net.generators.at[g, 'bus']
                for b in buses:
                    if str(bus_g) == str(b):
                        g_at_bus[season][b].append(g)
                        break
                        
        for s in storage_units:
            if s in net.storage_units.index:
                bus_s = net.storage_units.at[s, 'bus']
                for b in buses:
                    if str(bus_s) == str(b):
                        s_at_bus[season][b].append(s)
                        break
                        
        for l in lines:
            if l in net.lines.index:
                from_bus = net.lines.at[l, 'from_bus']
                to_bus = net.lines.at[l, 'to_bus']
                for b in buses:
                    if str(from_bus) == str(b):
                        lines_from[season][b].append(l)
                    if str(to_bus) == str(b):
                        lines_to[season][b].append(l)

    # ----------------------------------------------------------
    # Create variables ONCE for every (season, year) pair
    # ----------------------------------------------------------
    for s in seasons:
        net = integrated_network.season_networks[s]
        T = net.T

        for y in years:
            # ---------- generators ----------
            for g in generators:
                var = cp.Variable(T, nonneg=True, name=f"p_gen_{s}_{g}_{y}")
                p_gen[(g, y, s)] = var

            # ---------- lines ----------
            for l in lines:
                var = cp.Variable(T, name=f"p_line_{s}_{l}_{y}")
                p_line[(l, y, s)] = var

            # ---------- storage ----------
            for st in storage_units:
                p_charge[(st, y, s)] = cp.Variable(T, nonneg=True, name=f"p_charge_{s}_{st}_{y}")
                p_discharge[(st, y, s)] = cp.Variable(T, nonneg=True, name=f"p_discharge_{s}_{st}_{y}")
                soc[(st, y, s)] = cp.Variable(T, nonneg=True, name=f"soc_{s}_{st}_{y}")

    # Add capacity constraints for generators
    for (g, y, s), var in p_gen.items():
        net = integrated_network.season_networks[s]
        if g in net.generators.index:
            g_nom = net.generators.at[g, 'p_nom']
            g_type = net.generators.at[g, 'type']

            if g_type in ['wind', 'solar'] and \
               'p_max_pu' in net.generators_t and g in net.generators_t['p_max_pu']:
                prof = net.generators_t['p_max_pu'][g].values[:net.T]
                flat_constraints.append(
                    var <= cp.multiply(prof, g_nom) * gen_installed[(g, y)]
                )
            else:
                flat_constraints.append(
                    var <= g_nom * gen_installed[(g, y)]
                )
        else:
            flat_constraints.append(var <= 0)

    # Add constraints for line flows
    for (l, y, s), var in p_line.items():
        net = integrated_network.season_networks[s]
        if l in net.lines.index:
            line_cap = net.lines.at[l, 's_nom'] if hasattr(net.lines, 's_nom') else 0
            flat_constraints.append(cp.abs(var) <= line_cap)

    # Add constraints for storage
    for (st, y, s), charge_var in p_charge.items():
        discharge_var = p_discharge[(st, y, s)]
        soc_var = soc[(st, y, s)]
        net = integrated_network.season_networks[s]
        T = net.T
        
        if st in net.storage_units.index:
            s_p_nom = net.storage_units.at[st, 'p_nom']
            s_max_hours = net.storage_units.at[st, 'max_hours']
            eff_in = net.storage_units.at[st, 'efficiency_store']
            eff_out = net.storage_units.at[st, 'efficiency_dispatch']
            s_e_nom = s_p_nom * s_max_hours
        else:
            s_p_nom, s_e_nom, eff_in, eff_out = 0, 0, 1, 1
        
        # Capacity constraints using storage_installed
        flat_constraints.append(charge_var <= s_p_nom * storage_installed[(st, y)])
        flat_constraints.append(discharge_var <= s_p_nom * storage_installed[(st, y)])
        flat_constraints.append(soc_var <= s_e_nom * storage_installed[(st, y)])
        
        # SoC dynamics
        flat_constraints.append(
            soc_var[1:] == soc_var[:-1] + eff_in * charge_var[:-1] - (1.0/eff_out) * discharge_var[:-1]
        )
        flat_constraints.append(soc_var[0] == 0)
        # Relax final SoC constraint
        flat_constraints.append(soc_var[T-1] >= 0)
        flat_constraints.append(soc_var[T-1] <= s_e_nom * storage_installed[(st, y)] * 0.1)

    # Nodal power balance constraints
    load_growth_factors = integrated_network.load_growth if hasattr(integrated_network, 'load_growth') else {y: 1.0 for y in years}
    season_weights = integrated_network.season_weights if hasattr(integrated_network, 'season_weights') else {s: 1.0 for s in seasons}

    for s in seasons:
        net = integrated_network.season_networks[s]
        T = net.T
        for y in years:
            growth = load_growth_factors.get(y, 1.0)
            for b in buses:
                load_vec = cp.Constant(growth * bus_load_dict[s][b])
                
                gen_sum = cp.sum([p_gen[(g, y, s)] for g in g_at_bus[s][b]]) if g_at_bus[s][b] else 0
                st_net = cp.sum([p_discharge[(st, y, s)] - p_charge[(st, y, s)] for st in s_at_bus[s][b]]) if s_at_bus[s][b] else 0
                flow_out = cp.sum([p_line[(l, y, s)] for l in lines_from[s][b]]) if lines_from[s][b] else 0
                flow_in = cp.sum([p_line[(l, y, s)] for l in lines_to[s][b]]) if lines_to[s][b] else 0
                
                flat_constraints.append(
                    gen_sum + st_net + flow_in == load_vec + flow_out
                )

    # -- 4) Build the objective function: Operational cost + annualized capital cost
    # Operational cost with season weighting
    operational_obj = 0
    for (g, y, s), var in p_gen.items():
        if g in mc_g:  # Using pre-computed marginal costs
            net = integrated_network.season_networks[s]
            weight = season_weights.get(s, 0)
            mc = mc_g[g]
            # Remove discount factor for operational costs
            operational_obj += weight * mc * cp.sum(var)

    capital_obj = 0
    if first_network:
        # Process generator costs using the annuity approach
        for g in generators:
            if g not in first_network.generators.index: continue

            # Retrieve fixed parameters from the network data
            p_nom = first_network.generators.at[g, 'p_nom']
            capex_spec = first_network.generators.at[g, 'capex']   # $ / MW
            capex = capex_spec * p_nom                              # total $
            lifetime = first_network.generators.at[g, 'lifetime_years']
            discount_rate = first_network.generators.at[g, 'discount_rate']
            # Check if operating_costs exists, otherwise default to 0
            opex_fraction = 0.0
            if 'operating_costs' in first_network.generators.columns:
                opex_fraction = first_network.generators.at[g, 'operating_costs']
            
            # Default lifetime if needed
            if lifetime is None or pd.isna(lifetime) or lifetime <= 0:
                lifetime = 1

            # Compute the sum of discounted factors over the asset's lifetime
            discount_sum = compute_discount_sum(lifetime, discount_rate)
            # Compute the asset's NPV (here treating CAPEX and operating cost as cash outflows)
            npv = capex + (capex * opex_fraction * discount_sum)
            # Compute the capital recovery factor (CRF)
            crf = compute_crf(lifetime, discount_rate)
            # Annual cost (annuity) for the asset
            annual_asset_cost = npv * crf

            # For every planning year, if the generator is installed, add its full annuity cost (without additional discounting)
            for y in years:
                capital_obj += annual_asset_cost * gen_installed[(g, y)]

        # Process storage costs similarly:
        for s in storage_units:
            if s not in first_network.storage_units.index: continue

            # Retrieve fixed parameters from the network data
            p_nom = first_network.storage_units.at[s, 'p_nom']
            capex_spec = first_network.storage_units.at[s, 'capex']   # $ / MW
            capex = capex_spec * p_nom                              # total $
            lifetime_s = first_network.storage_units.at[s, 'lifetime_years']
            discount_rate_s = first_network.storage_units.at[s, 'discount_rate']
            # Check if operating_costs exists, otherwise default to 0
            opex_fraction_s = 0.0
            if 'operating_costs' in first_network.storage_units.columns:
                opex_fraction_s = first_network.storage_units.at[s, 'operating_costs']
            
            # Default lifetime if needed
            if lifetime_s is None or pd.isna(lifetime_s) or lifetime_s <= 0:
                lifetime_s = 1

            # Compute the sum of discounted factors over the asset's lifetime
            discount_sum_s = compute_discount_sum(lifetime_s, discount_rate_s)
            # Compute the asset's NPV (here treating CAPEX and operating cost as cash outflows)
            npv_s = capex + (capex * opex_fraction_s * discount_sum_s)
            # Compute the capital recovery factor (CRF)
            crf_s = compute_crf(lifetime_s, discount_rate_s)
            # Annual cost (annuity) for the asset
            annual_asset_cost_s = npv_s * crf_s

            # For every planning year, if the storage is installed, add its full annuity cost (without additional discounting)
            for y in years:
                capital_obj += annual_asset_cost_s * storage_installed[(s, y)]

    total_cost = operational_obj + capital_obj
    objective = cp.Minimize(total_cost)
    
    all_constraints = global_constraints + flat_constraints

    return {
        'objective': objective,
        'constraints': all_constraints,
        'variables': {
            'gen_build': gen_build,             # Return new build vars
            'storage_build': storage_build,     # Return new build vars
            'gen_installed': gen_installed,     # Return installed vars
            'storage_installed': storage_installed, # Return installed vars
            'p_gen': p_gen,
            'p_line': p_line,
            'p_charge': p_charge,
            'p_discharge': p_discharge,
            'soc': soc
        },
        'years': years,
        'seasons': seasons
    }

def investement_multi(integrated_network, solver_options=None):
    """
    Solve the integrated multi-year problem using the build/installed formulation:
      1) Create the problem
      2) Solve with CPLEX
      3) Extract numeric variable values (including build vars)
      4) Store results in integrated_network.integrated_results
    """
    if solver_options is None:
        solver_options = {}
    
    years = integrated_network.years
    seasons = integrated_network.seasons
    first_network = list(integrated_network.season_networks.values())[0] if seasons else None
    logger.info(f"Solving multi-year investment: {len(seasons)} seasons, {len(years)} years using build/installed formulation.")

    # 1) Create problem
    problem_dict = dcopf(integrated_network)
    prob = cp.Problem(problem_dict['objective'], problem_dict['constraints'])

    # 2) Solve with CPLEX
    cplex_params = {'threads': 10, 'timelimit': 18*60, 'mip.tolerances.mipgap': 0.01}
    cplex_params.update(solver_options)

    try:
        logger.info("Solving with CPLEX...")
        # Increase MIP tolerances for potentially challenging problems
        cplex_params.update({'mip.tolerances.absmipgap': 1.0})
        prob.solve(solver=cp.CPLEX, verbose=True, cplex_params=cplex_params)
    except Exception as e:
        logger.error(f"Solver failed: {e}")
        # Check if the error message contains DCP details
        if "Problem does not follow DCP rules" in str(e):
            logger.error("DCP Error Details:")
            # (CVXPY often prints DCP errors to stdout/stderr, check console)
            # Extracting specific details programmatically can be complex
            pass 
        integrated_network.integrated_results = {
            'status': 'failed',
            'value': None,
            'success': False,
            'variables': {}
        }
        return integrated_network.integrated_results

    status = prob.status
    objective_value = prob.value if status in ("optimal", "optimal_inaccurate") else None

    if status not in ("optimal", "optimal_inaccurate"):
        logger.warning(f"Solve ended with status {status}. Objective value: {objective_value}")
        # Attempt to analyze infeasibility if CPLEX provides details
        # (This requires specific CPLEX API calls not directly available via CVXPY)
        integrated_network.integrated_results = {
            'status': status,
            'value': objective_value,
            'success': False,
            'variables': {}
        }
        return integrated_network.integrated_results
    else:
        logger.info(f"Solve successful with status {status}. Objective value: {objective_value:.2f}")


    # 3) Extract variable values
    result_vars = {}
    var_values = {}
    problem_vars = problem_dict['variables']
    
    # Helper to safely get variable value
    def get_val(var):
        try:
            return float(var.value) if var.value is not None else 0.0
        except (AttributeError, TypeError, ValueError):
            return 0.0 # Default to 0 if value is invalid or missing

    # Extract build variables
    for k, var in problem_vars['gen_build'].items():
        var_values[k] = get_val(var)
        result_vars[f"gen_build_{k[0]}_{k[1]}"] = var_values[k]
    for k, var in problem_vars['storage_build'].items():
        var_values[k] = get_val(var)
        result_vars[f"storage_build_{k[0]}_{k[1]}"] = var_values[k]

    # Extract installed variables
    for k, var in problem_vars['gen_installed'].items():
        var_values[k] = get_val(var)
        result_vars[f"gen_installed_{k[0]}_{k[1]}"] = var_values[k]
    for k, var in problem_vars['storage_installed'].items():
        var_values[k] = get_val(var)
        result_vars[f"storage_installed_{k[0]}_{k[1]}"] = var_values[k]

    # Extract seasonal dispatch variables using the flat dictionaries
    for var_type, var_dict in [('p_gen', problem_vars['p_gen']), 
                               ('p_line', problem_vars['p_line']), 
                               ('p_charge', problem_vars['p_charge']), 
                               ('p_discharge', problem_vars['p_discharge']), 
                               ('soc', problem_vars['soc'])]:
        for key, var_vec in var_dict.items():
            # key is (asset_id, year, season)
            asset_id, year, season = key
            # Extract values safely
            try:
                vals = var_vec.value
                if vals is None:
                    net = integrated_network.season_networks[season]
                    T = net.T
                    vals = np.zeros(T)
                elif not isinstance(vals, np.ndarray):
                    net = integrated_network.season_networks[season]
                    T = net.T
                    vals = np.full(T, float(vals))
            except (AttributeError, TypeError, ValueError):
                net = integrated_network.season_networks[season]
                T = net.T
                logger.warning(f"Could not get value for {var_type} {key}. Defaulting to zeros.")
                vals = np.zeros(T)

            # Store individual time step values with padded hour format
            for t in range(len(vals)):
                result_vars[f"{var_type}_{season}_{asset_id}_{year}_{t+1:03d}"] = float(vals[t])

    # 4) Save solution
    integrated_network.integrated_results = {
        'status': status,
        'value': objective_value,
        'success': True,
        'variables': result_vars
    }
    
    # Log summary using extracted values from result_vars
    num_gen_built = sum(1 for k, v in result_vars.items() if k.startswith('gen_build_') and v > 0.5)
    num_stor_built = sum(1 for k, v in result_vars.items() if k.startswith('storage_build_') and v > 0.5)
    logger.info(f"Total generator build decisions: {num_gen_built}")
    logger.info(f"Total storage build decisions: {num_stor_built}")

    # Detailed installation analysis using 'installed' variables
    logger.info("Analyzing installation patterns (based on 'installed' status):")
    gen_installations = {}
    storage_installations = {}
    
    # Use result_vars which contains the extracted float values
    for k, v in result_vars.items():
        if v > 0.5: # Use a threshold for binary check
            if k.startswith('gen_installed_'):
                parts = k.split('_')
                if len(parts) == 4: # Check format: gen_installed_ID_YEAR
                   try:
                       g = parts[2]
                       y = int(parts[3])
                       if g not in gen_installations: gen_installations[g] = []
                       gen_installations[g].append(y)
                   except (IndexError, ValueError):
                       logger.warning(f"Could not parse key: {k}")
            elif k.startswith('storage_installed_'):
                parts = k.split('_')
                if len(parts) == 4:
                    try:
                        s = parts[2]
                        y = int(parts[3])
                        if s not in storage_installations: storage_installations[s] = []
                        storage_installations[s].append(y)
                    except (IndexError, ValueError):
                        logger.warning(f"Could not parse key: {k}")
    
    # Function to get lifetime safely
    def get_lifetime(asset_id, asset_type):
        df = first_network.generators if asset_type == 'gen' else first_network.storage_units
        try:
            # Handle potential integer/string mismatch for index lookup
            if asset_id in df.index:
                return df.at[asset_id, 'lifetime_years']
            elif str(asset_id) in df.index:
                 return df.at[str(asset_id), 'lifetime_years']
            elif int(asset_id) in df.index:
                 return df.at[int(asset_id), 'lifetime_years']
            else:
                return "unknown"
        except (KeyError, ValueError, TypeError):
            return "unknown"

    logger.info("Generator installation patterns:")
    for g, years_list in gen_installations.items():
        lifetime = get_lifetime(g, 'gen')
        years_str = ", ".join(str(y) for y in sorted(list(set(years_list)))) # Ensure unique and sorted
        logger.info(f"  Generator {g} (lifetime={lifetime}): installed in years [{years_str}]")
        
    logger.info("Storage installation patterns:")
    for s, years_list in storage_installations.items():
        lifetime = get_lifetime(s, 'storage')
        years_str = ", ".join(str(y) for y in sorted(list(set(years_list)))) # Ensure unique and sorted
        logger.info(f"  Storage {s} (lifetime={lifetime}): installed in years [{years_str}]")

    return integrated_network.integrated_results