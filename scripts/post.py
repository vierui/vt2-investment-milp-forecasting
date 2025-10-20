#!/usr/bin/env python3
"""
Post-processing module for the simplified multi-year approach.

We demonstrate:
  - Implementation plan generation
  - Possibly cost breakdown if desired
  - Visualization of profiles for different generator types
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def key(var, season, asset, year, hour):
    """Return the canonical optimisation‑variable key (3‑digit hour)."""
    return f"{var}_{season}_{asset}_{year}_{hour:03d}"

def test_key_lookup():
    k = key('p_gen', 'winter', 'G1', 3, 7)
    assert k == 'p_gen_winter_G1_3_007'

# A custom JSON encoder to handle e.g. numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def generate_implementation_plan(integrated_network, output_dir="results"):
    """
    Build a simple plan describing in which year each generator or storage 
    is actually installed (binary=1).
    Then optionally we can parse any dispatch results if needed.

    integrated_network: 
      - Should have an 'integrated_results' dict with a 'variables' sub-dict 
        containing numeric values (not CVXPY variables).

    Returns: A dictionary summarizing the plan, also saved as JSON in output_dir.
    """

    if not hasattr(integrated_network, 'integrated_results') or integrated_network.integrated_results is None:
        print("[post.py] No integrated_results found on integrated_network.")
        return {}

    result = integrated_network.integrated_results
    if 'variables' not in result:
        print("[post.py] integrated_results has no 'variables'.")
        return {}

    var_dict = result['variables']

    # ------------------------------------------------------------------
    # Identify build years   (🔄 replacing old 'installed' logic)
    # ------------------------------------------------------------------

    gen_build = {}
    storage_build = {}

    for key, value in var_dict.items():
        if value <= 0.5:          # skip zeros
            continue

        if key.startswith('gen_build_'):
            # key format: gen_build_<ID>_<YEAR>
            _, _, g_id, y = key.split('_')
            gen_build.setdefault(g_id, []).append(int(y))

        elif key.startswith('storage_build_'):
            # key format: storage_build_<ID>_<YEAR>
            _, _, s_id, y = key.split('_')
            storage_build.setdefault(s_id, []).append(int(y))

    # ------------------------------------------------------------------
    # Assemble plan dictionary
    # ------------------------------------------------------------------
    plan = {
        'years': integrated_network.years,
        'generators': {g: {'years_installed': sorted(ys)}
                       for g, ys in gen_build.items()},
        'storage':    {s: {'years_installed': sorted(ys)}
                       for s, ys in storage_build.items()},
        'objective_value': result.get('value', 0),
        'status': result.get('status', 'unknown')
    }

    # Save to JSON
    os.makedirs(output_dir, exist_ok=True)
    plan_file = os.path.join(output_dir, "implementation_plan.json")
    try:
        with open(plan_file, "w") as f:
            json.dump(plan, f, indent=2, cls=NumpyEncoder)
        print(f"[post.py] Implementation plan saved to {plan_file}")
    except Exception as e:
        print(f"[post.py] Error saving plan JSON: {e}")

    return plan


def save_cost_breakdown(integrated_network, output_dir="results"):
    """
    Optional: If you store cost breakdown by year or season, you can produce 
    a separate JSON. Here is a skeleton that you can adapt.
    """

    if not hasattr(integrated_network, 'integrated_results'):
        print("[post.py] No integrated_results to derive cost breakdown.")
        return {}

    result = integrated_network.integrated_results
    # If you had separate operational/capital cost in your solution, 
    # you'd read them from the model or parse from the variables. 
    # Suppose you do it manually or it's not separated. Just store total.
    cost_info = {
        'objective': result.get('value', None),
        'status': result.get('status', 'unknown')
        # Could add more details if your solver tracked them
    }

    cost_file = os.path.join(output_dir, "cost_breakdown.json")
    try:
        with open(cost_file, "w") as f:
            json.dump(cost_info, f, indent=2, cls=NumpyEncoder)
        print(f"[post.py] Cost breakdown saved to {cost_file}")
    except Exception as e:
        print(f"[post.py] Error saving cost breakdown: {e}")

    return cost_info


def plot_seasonal_profiles(integrated_network, output_dir="results"):
    """
    Create and save plots showing the total profiles for solar, wind, thermal, and loads
    for each season (particularly summer and winter).
    
    Args:
        integrated_network: IntegratedNetwork object with season networks
        output_dir: Directory to save output plots
    
    Returns:
        Dictionary with plot filenames
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    plot_files = {}
    
    # Seasons to plot (focus on summer and winter)
    seasons_to_plot = ['summer', 'winter']
    if 'spri_autu' in integrated_network.seasons:
        seasons_to_plot.append('spri_autu')
    
    # For each season, collect and plot the profiles
    for season in seasons_to_plot:
        if season not in integrated_network.season_networks:
            print(f"[post.py] Season {season} not found in network, skipping")
            continue
            
        network = integrated_network.season_networks[season]
        hours = range(1, network.T + 1)
        
        # Create a new figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f"Resource Profiles for {season.capitalize()} Season", fontsize=16)
        
        # Plot 1: Generator availability by type
        # Collect data for wind, solar, and thermal generators
        wind_total = np.zeros(network.T)
        solar_total = np.zeros(network.T)
        thermal_cap = 0
        
        # Get total capacity by generator type
        for gen_id in network.generators.index:
            gen_type = network.generators.at[gen_id, 'type']
            gen_cap = network.generators.at[gen_id, 'p_nom']
            
            if gen_type == 'wind' and gen_id in network.generators_t['p_max_pu']:
                profile = network.generators_t['p_max_pu'][gen_id].values
                wind_total += profile
            elif gen_type == 'solar' and gen_id in network.generators_t['p_max_pu']:
                profile = network.generators_t['p_max_pu'][gen_id].values
                solar_total += profile
            elif gen_type == 'thermal':
                thermal_cap += gen_cap
        
        # Plot generator availability
        ax1.plot(hours, wind_total, label='Wind', color='blue', linewidth=2)
        ax1.plot(hours, solar_total, label='Solar', color='orange', linewidth=2)
        ax1.axhline(y=thermal_cap, label='Thermal', color='brown', linestyle='-', linewidth=2)
        
        ax1.set_ylabel('Available Capacity (MW)')
        ax1.set_title('Generator Availability Profiles')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Format y-axis to show integers
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Plot 2: Total load profile
        # Initialize total_load array
        total_load = np.zeros(network.T)
        
        # Collect total load for each hour
        for bus in network.buses.index:
            # Get the load at this bus for each hour
            bus_load = np.zeros(network.T)
            for load_id in network.loads.index:
                if network.loads.at[load_id, 'bus'] == bus:
                    if load_id in network.loads_t['p']:
                        bus_load += network.loads_t['p'][load_id].values
                    else:
                        bus_load += np.ones(network.T) * network.loads.at[load_id, 'p_mw']
            total_load += bus_load
        
        # Plot total load
        ax2.plot(hours, total_load, label='Total Load', color='red', linewidth=2)
        ax2.set_xlabel('Hour')
        ax2.set_ylabel('Load (MW)')
        ax2.set_title('Total System Load Profile')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Set common x-axis limits and labels
        ax1.set_xlim(1, network.T)
        ax2.set_xlim(1, network.T)
        
        # Finalize and save the plot
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the title
        plot_file = os.path.join(output_dir, f"{season}_profiles.png")
        plt.savefig(plot_file, dpi=300)
        plt.close(fig)
        
        plot_files[season] = plot_file
        print(f"[post.py] Created profile plot for {season} at {plot_file}")
        
    # Create a combined plot comparing seasons
    if len(seasons_to_plot) > 1:
        # Create a figure with 4 subplots
        fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        fig.suptitle("Comparison of Seasonal Resource Profiles", fontsize=16)
        
        # Dictionary to store data for each season
        season_data = {}
        
        # Collect data for all seasons
        for season in seasons_to_plot:
            if season not in integrated_network.season_networks:
                continue
                
            network = integrated_network.season_networks[season]
            wind_total = np.zeros(network.T)
            solar_total = np.zeros(network.T)
            thermal_cap = 0
            total_load = np.zeros(network.T)
            
            # Collect generator data
            for gen_id in network.generators.index:
                gen_type = network.generators.at[gen_id, 'type']
                gen_cap = network.generators.at[gen_id, 'p_nom']
                
                if gen_type == 'wind' and gen_id in network.generators_t['p_max_pu']:
                    profile = network.generators_t['p_max_pu'][gen_id].values
                    wind_total += profile
                elif gen_type == 'solar' and gen_id in network.generators_t['p_max_pu']:
                    profile = network.generators_t['p_max_pu'][gen_id].values
                    solar_total += profile
                elif gen_type == 'thermal':
                    thermal_cap += gen_cap
            
            # Collect load data
            for bus in network.buses.index:
                bus_load = np.zeros(network.T)
                for load_id in network.loads.index:
                    if network.loads.at[load_id, 'bus'] == bus:
                        if load_id in network.loads_t['p']:
                            bus_load += network.loads_t['p'][load_id].values
                        else:
                            bus_load += np.ones(network.T) * network.loads.at[load_id, 'p_mw']
            
            season_data[season] = {
                'wind': wind_total,
                'solar': solar_total,
                'thermal': thermal_cap,
                'load': total_load,
                'hours': range(1, network.T + 1)
            }
        
        # Plot wind profiles for all seasons
        for season in seasons_to_plot:
            if season in season_data:
                axs[0].plot(season_data[season]['hours'], season_data[season]['wind'], 
                           label=season.capitalize(), linewidth=2)
        axs[0].set_title('Wind Availability Profiles')
        axs[0].set_ylabel('Available Capacity (MW)')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)
        
        # Plot solar profiles for all seasons
        for season in seasons_to_plot:
            if season in season_data:
                axs[1].plot(season_data[season]['hours'], season_data[season]['solar'], 
                           label=season.capitalize(), linewidth=2)
        axs[1].set_title('Solar Availability Profiles')
        axs[1].set_ylabel('Available Capacity (MW)')
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)
        
        # Plot thermal capacity for all seasons
        for season in seasons_to_plot:
            if season in season_data:
                axs[2].axhline(y=season_data[season]['thermal'], 
                              label=f"{season.capitalize()} ({season_data[season]['thermal']} MW)", 
                              linewidth=2)
        axs[2].set_title('Thermal Capacity')
        axs[2].set_ylabel('Capacity (MW)')
        axs[2].legend()
        axs[2].grid(True, alpha=0.3)
        
        # Plot load profiles for all seasons
        for season in seasons_to_plot:
            if season in season_data:
                axs[3].plot(season_data[season]['hours'], season_data[season]['load'], 
                           label=season.capitalize(), linewidth=2)
        axs[3].set_title('Total System Load Profiles')
        axs[3].set_xlabel('Hour')
        axs[3].set_ylabel('Load (MW)')
        axs[3].legend()
        axs[3].grid(True, alpha=0.3)
        
        # Set x-axis limits for all subplots
        for ax in axs:
            if len(season_data) > 0:
                first_season = next(iter(season_data))
                ax.set_xlim(1, len(season_data[first_season]['hours']))
        
        # Finalize and save the combined plot
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the title
        combined_plot_file = os.path.join(output_dir, "seasonal_profiles_comparison.png")
        plt.savefig(combined_plot_file, dpi=300)
        plt.close(fig)
        
        plot_files['comparison'] = combined_plot_file
        print(f"[post.py] Created combined seasonal profile comparison at {combined_plot_file}")
    
    return plot_files


def plot_annual_load_growth(integrated_network, output_dir="results"):
    """
    Create and save a plot showing how the total system load grows over the planning horizon
    based on the load growth factors defined in analysis.json.
    
    Args:
        integrated_network: IntegratedNetwork object with season networks and load growth factors
        output_dir: Directory to save output plot
    
    Returns:
        Path to the generated plot file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if load growth factors are available
    if not hasattr(integrated_network, 'load_growth') or not integrated_network.load_growth:
        print("[post.py] No load growth factors found on integrated_network")
        return None
    
    # Get years and load growth factors
    years = integrated_network.years
    load_growth_factors = integrated_network.load_growth
    
    # Choose a representative season to extract the base load
    # We'll use the first season in the list for simplicity
    if not integrated_network.seasons or not integrated_network.season_networks:
        print("[post.py] No seasons found in integrated_network")
        return None
    
    representative_season = integrated_network.seasons[0]
    network = integrated_network.season_networks[representative_season]
    
    # Calculate base load (sum of load across all buses for the representative period)
    base_load_by_bus = {}
    total_base_load = 0
    
    for bus in network.buses.index:
        # Get the load at this bus for each hour
        bus_load = np.zeros(network.T)
        for load_id in network.loads.index:
            if network.loads.at[load_id, 'bus'] == bus:
                if 'p' in network.loads_t and load_id in network.loads_t['p']:
                    bus_load += network.loads_t['p'][load_id].values[:network.T]
                else:
                    bus_load += np.ones(network.T) * network.loads.at[load_id, 'p_mw']
        
        # Store the bus load for future reference
        base_load_by_bus[bus] = bus_load
        
        # Add to total base load
        total_base_load += np.sum(bus_load)
    
    # Calculate total load for each year by applying growth factors
    total_load_by_year = {}
    avg_load_by_year = {}
    
    for y in years:
        growth_factor = load_growth_factors.get(y, 1.0)
        total_load_by_year[y] = total_base_load * growth_factor
        avg_load_by_year[y] = total_load_by_year[y] / network.T  # Average MW over the period
    
    # Create a plot of annual load growth
    plt.figure(figsize=(10, 6))
    
    # Plot total load (MWh over representative period)
    plt.subplot(2, 1, 1)
    plt.plot(years, [total_load_by_year[y] for y in years], 'o-', linewidth=2, color='blue')
    plt.title('Annual Load Growth - Total Energy')
    plt.xlabel('Year')
    plt.ylabel('Total Load (MWh over representative period)')
    plt.grid(True, alpha=0.3)
    
    # Plot average load (MW)
    plt.subplot(2, 1, 2)
    plt.plot(years, [avg_load_by_year[y] for y in years], 'o-', linewidth=2, color='red')
    plt.title('Annual Load Growth - Average Power')
    plt.xlabel('Year')
    plt.ylabel('Average Load (MW)')
    plt.grid(True, alpha=0.3)
    
    # Add the growth factor as text annotations for a few selected years
    # (first, middle, and last year)
    selected_years = [years[0], years[len(years)//2], years[-1]]
    for y in selected_years:
        growth_factor = load_growth_factors.get(y, 1.0)
        plt.annotate(f'{growth_factor:.2f}x',
                     xy=(y, avg_load_by_year[y]),
                     xytext=(0, 10),
                     textcoords='offset points',
                     ha='center')
    
    # Adjust layout and save the plot
    plt.tight_layout()
    plot_file = os.path.join(output_dir, "annual_load_growth.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    
    print(f"[post.py] Annual load growth plot saved to {plot_file}")
    
    # Also print the data as a table for reference
    print("\nAnnual Load Growth Summary:")
    print("Year | Growth Factor | Total Load (MWh) | Average Load (MW)")
    print("-" * 60)
    for y in years:
        growth_factor = load_growth_factors.get(y, 1.0)
        print(f"{y:4d} | {growth_factor:13.4f} | {total_load_by_year[y]:15.2f} | {avg_load_by_year[y]:16.2f}")
    
    return plot_file


def plot_generation_mix(integrated_network, output_dir="results"):
    """
    Create and save plots showing how the load is met by different generators
    and storage units for winter and summer seasons in years 1 and 10.
    
    The plot shows stacked area charts representing the generation mix hour by hour.
    
    Args:
        integrated_network: IntegratedNetwork object with season networks and optimization results
        output_dir: Directory to save output plots
    
    Returns:
        Dictionary with plot filenames
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    plot_files = {}
    
    # Check if we have optimization results
    if not hasattr(integrated_network, 'integrated_results'):
        print("[post.py] No optimization results found")
        return plot_files
        
    result = integrated_network.integrated_results
    if 'variables' not in result or not result['variables']:
        print("[post.py] Error: Optimization results not found or empty.")
        return plot_files
    
    # Get years - we want to compare first and last year
    if not integrated_network.years or len(integrated_network.years) < 2:
        print("[post.py] Error: Need at least 2 years in the planning horizon.")
        return plot_files
        
    first_year = integrated_network.years[0]
    last_year = integrated_network.years[-1]
    
    # Seasons to plot
    seasons_to_plot = ['winter', 'summer']
    
    # Get load growth factors
    load_growth_factors = getattr(integrated_network, 'load_growth', {})
    growth_factor_first_year = load_growth_factors.get(first_year, 1.0)
    growth_factor_last_year = load_growth_factors.get(last_year, 1.0)
    
    # Function to generate colors based on ID
    def get_color_map(asset_ids):
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        
        # Create a colormap with distinct colors
        color_map = {}
        if asset_ids:
            cmap = cm.get_cmap('tab20', len(asset_ids))
            for i, asset_id in enumerate(asset_ids):
                color_map[asset_id] = cmap(i)
        return color_map
    
    # For each season
    for season in seasons_to_plot:
        if season not in integrated_network.season_networks:
            print(f"[post.py] Season {season} not found in network, skipping")
            continue
            
        network = integrated_network.season_networks[season]
        hours = range(1, network.T + 1)
        
        # Get all generator IDs
        all_gens = list(network.generators.index)
        
        # Get all storage unit IDs
        all_storage = list(network.storage_units.index)
        
        # Create consistent color maps for assets
        gen_colors = get_color_map(all_gens)
        storage_colors = get_color_map(all_storage)
        
        # Create arrays to store dispatch by generator ID for each year
        dispatch_y1 = {gen_id: np.zeros(network.T) for gen_id in all_gens}
        dispatch_y10 = {gen_id: np.zeros(network.T) for gen_id in all_gens}
        
        # Create arrays to store discharge and charge by storage ID for each year
        storage_discharge_y1 = {stor_id: np.zeros(network.T) for stor_id in all_storage}
        storage_charge_y1 = {stor_id: np.zeros(network.T) for stor_id in all_storage}
        storage_discharge_y10 = {stor_id: np.zeros(network.T) for stor_id in all_storage}
        storage_charge_y10 = {stor_id: np.zeros(network.T) for stor_id in all_storage}
        
        # Calculate total load for both years
        total_load_y1 = np.zeros(network.T)
        total_load_y10 = np.zeros(network.T)
        
        # Get the load at each bus for each hour
        for bus in network.buses.index:
            bus_load = np.zeros(network.T)
            for load_id in network.loads.index:
                if network.loads.at[load_id, 'bus'] == bus:
                    if 'p' in network.loads_t and load_id in network.loads_t['p']:
                        bus_load += network.loads_t['p'][load_id].values[:network.T]
                    else:
                        bus_load += np.ones(network.T) * network.loads.at[load_id, 'p_mw']
            
            # Apply load growth factors
            total_load_y1 += bus_load * growth_factor_first_year
            total_load_y10 += bus_load * growth_factor_last_year
        
        # Sum up generator dispatch for each hour
        for hour in range(1, network.T + 1):
            # Year 1
            for gen_id in all_gens:
                dispatch_key = key('p_gen', season, gen_id, first_year, hour)
                dispatch = result['variables'].get(dispatch_key, 0)
                dispatch_y1[gen_id][hour-1] = dispatch
                
            # Year 10
            for gen_id in all_gens:
                dispatch_key = key('p_gen', season, gen_id, last_year, hour)
                dispatch = result['variables'].get(dispatch_key, 0)
                dispatch_y10[gen_id][hour-1] = dispatch
        
        # Get storage charge and discharge for each storage unit
        for hour in range(1, network.T + 1):
            # Year 1
            for stor_id in all_storage:
                charge_key = key('p_charge', season, stor_id, first_year, hour)
                discharge_key = key('p_discharge', season, stor_id, first_year, hour)
                
                charge = result['variables'].get(charge_key, 0)
                discharge = result['variables'].get(discharge_key, 0)
                
                # Negative because charging reduces net generation
                storage_charge_y1[stor_id][hour-1] = -charge
                storage_discharge_y1[stor_id][hour-1] = discharge
            
            # Year 10
            for stor_id in all_storage:
                charge_key = key('p_charge', season, stor_id, last_year, hour)
                discharge_key = key('p_discharge', season, stor_id, last_year, hour)
                
                charge = result['variables'].get(charge_key, 0)
                discharge = result['variables'].get(discharge_key, 0)
                
                # Negative because charging reduces net generation
                storage_charge_y10[stor_id][hour-1] = -charge
                storage_discharge_y10[stor_id][hour-1] = discharge
        
        # Calculate total generation and storage for verification
        total_gen_y1 = np.zeros(network.T)
        total_gen_y10 = np.zeros(network.T)
        
        # Sum all generation
        for gen_id in all_gens:
            total_gen_y1 += dispatch_y1[gen_id]
            total_gen_y10 += dispatch_y10[gen_id]
        
        # Sum all storage discharge and charge
        total_storage_discharge_y1 = np.zeros(network.T)
        total_storage_charge_y1 = np.zeros(network.T)
        total_storage_discharge_y10 = np.zeros(network.T)
        total_storage_charge_y10 = np.zeros(network.T)
        
        for stor_id in all_storage:
            total_storage_discharge_y1 += storage_discharge_y1[stor_id]
            total_storage_charge_y1 += storage_charge_y1[stor_id]
            total_storage_discharge_y10 += storage_discharge_y10[stor_id]
            total_storage_charge_y10 += storage_charge_y10[stor_id]
            
        # VERIFICATION: Check if load is always met using the actual optimizer variables
        print(f"\nLOAD BALANCE VERIFICATION FOR {season.upper()} SEASON:")
        
        # Year 1 verification
        total_with_storage_y1 = total_gen_y1 + total_storage_discharge_y1 + total_storage_charge_y1  # Charge is already negative
        
        # Calculate the maximum absolute error
        max_error_y1 = np.max(np.abs(total_with_storage_y1 - total_load_y1))
        avg_error_y1 = np.mean(np.abs(total_with_storage_y1 - total_load_y1))
        
        print(f"  Year {first_year} - Maximum load balance error: {max_error_y1:.6f} MW")
        print(f"  Year {first_year} - Average load balance error: {avg_error_y1:.6f} MW")
        
        if max_error_y1 > 0.01:  # If error is more than 0.01 MW
            print("  CAUTION: Load might not be fully met in some hours (Year 1)")
            # Print the hours with largest errors
            worst_hours_y1 = np.argsort(np.abs(total_with_storage_y1 - total_load_y1))[-5:]
            print("  Hours with largest imbalance (Year 1):")
            for h in worst_hours_y1:
                print(f"    Hour {h+1}: Load={total_load_y1[h]:.2f} MW, Generation+Storage={total_with_storage_y1[h]:.2f} MW, Diff={total_with_storage_y1[h]-total_load_y1[h]:.2f} MW")
        else:
            print(f"  Year {first_year} - Load is met for all hours within tolerance")
            
        # Year 10 verification
        total_with_storage_y10 = total_gen_y10 + total_storage_discharge_y10 + total_storage_charge_y10  # Charge is already negative
        
        # Calculate the maximum absolute error
        max_error_y10 = np.max(np.abs(total_with_storage_y10 - total_load_y10))
        avg_error_y10 = np.mean(np.abs(total_with_storage_y10 - total_load_y10))
        
        print(f"  Year {last_year} - Maximum load balance error: {max_error_y10:.6f} MW")
        print(f"  Year {last_year} - Average load balance error: {avg_error_y10:.6f} MW")
        
        if max_error_y10 > 0.01:  # If error is more than 0.01 MW
            print("  CAUTION: Load might not be fully met in some hours (Year 10)")
            # Print the hours with largest errors
            worst_hours_y10 = np.argsort(np.abs(total_with_storage_y10 - total_load_y10))[-5:]
            print("  Hours with largest imbalance (Year 10):")
            for h in worst_hours_y10:
                print(f"    Hour {h+1}: Load={total_load_y10[h]:.2f} MW, Generation+Storage={total_with_storage_y10[h]:.2f} MW, Diff={total_with_storage_y10[h]-total_load_y10[h]:.2f} MW")
        else:
            print(f"  Year {last_year} - Load is met for all hours within tolerance")
        
        # Create stacked area chart for Year 1
        plt.figure(figsize=(14, 8))
        
        # Plot load as a reference area (filled to make it clear the load must be met)
        plt.fill_between(hours, 0, total_load_y1, color='lightgray', alpha=0.3)
        
        # Plot each generator and storage discharge separately
        gen_data_y1 = []
        labels_y1 = []
        colors_y1 = []
        
        # Add generators
        for gen_id in all_gens:
            if np.any(dispatch_y1[gen_id] > 0):  # Only include generators with dispatch
                gen_data_y1.append(dispatch_y1[gen_id])
                labels_y1.append(f"Gen {gen_id}")
                colors_y1.append(gen_colors[gen_id])
        
        # Add storage discharge
        for stor_id in all_storage:
            if np.any(storage_discharge_y1[stor_id] > 0):  # Only add if there's actual discharge
                gen_data_y1.append(storage_discharge_y1[stor_id])
                labels_y1.append(f"Storage {stor_id} (Discharge)")
                colors_y1.append(storage_colors[stor_id])
        
        # Plot the generation stack
        plt.stackplot(hours, 
                     *gen_data_y1,
                     labels=labels_y1,
                     colors=colors_y1,
                     alpha=0.7)
        
        # Plot load as a solid line for comparison
        plt.plot(hours, total_load_y1, 'r-', linewidth=2.5, label='Load')
        
        # Plot storage charging individually
        for stor_id in all_storage:
            charge = storage_charge_y1[stor_id]
            if np.any(charge < 0):  # Only plot if there's actual charging (note: charge values are negative)
                charge_color = storage_colors[stor_id]
                # Make charge color slightly different than discharge color for the same storage
                charge_color = [min(1.0, c * 0.8) for c in charge_color[:3]] + [0.3]  # Adjust color and alpha
                plt.fill_between(hours, 0, charge, color=charge_color, label=f"Storage {stor_id} (Charge)")
        
        # Plot the sum of generation (should exactly match load)
        plt.plot(hours, total_with_storage_y1, 'k--', linewidth=1.5, label='Net Generation')
        
        plt.title(f'Generation Mix for {season.capitalize()} Season - Year {first_year}', fontsize=16)
        plt.xlabel('Hour', fontsize=12)
        plt.ylabel('Power (MW)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right', fontsize=10)
        
        # Add a text box with verification results
        plt.text(0.02, 0.02, f"Maximum balance error: {max_error_y1:.6f} MW\nAverage balance error: {avg_error_y1:.6f} MW", 
                transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        
        # Save the plot
        y1_plot_file = os.path.join(output_dir, f"{season}_mix_year{first_year}.png")
        plt.savefig(y1_plot_file, dpi=300)
        plt.close()
        
        plot_files[f"{season}_year{first_year}"] = y1_plot_file
        print(f"[post.py] Created generation mix plot for {season} year {first_year} at {y1_plot_file}")
        
        # Create stacked area chart for Year 10
        plt.figure(figsize=(14, 8))
        
        # Plot load as a reference area (filled to make it clear the load must be met)
        plt.fill_between(hours, 0, total_load_y10, color='lightgray', alpha=0.3)
        
        # Plot each generator and storage discharge separately
        gen_data_y10 = []
        labels_y10 = []
        colors_y10 = []
        
        # Add generators
        for gen_id in all_gens:
            if np.any(dispatch_y10[gen_id] > 0):  # Only include generators with dispatch
                gen_data_y10.append(dispatch_y10[gen_id])
                labels_y10.append(f"Gen {gen_id}")
                colors_y10.append(gen_colors[gen_id])
        
        # Add storage discharge
        for stor_id in all_storage:
            if np.any(storage_discharge_y10[stor_id] > 0):  # Only add if there's actual discharge
                gen_data_y10.append(storage_discharge_y10[stor_id])
                labels_y10.append(f"Storage {stor_id} (Discharge)")
                colors_y10.append(storage_colors[stor_id])
        
        # Plot the generation stack
        plt.stackplot(hours, 
                     *gen_data_y10,
                     labels=labels_y10,
                     colors=colors_y10,
                     alpha=0.7)
        
        # Plot load as a solid line for comparison
        plt.plot(hours, total_load_y10, 'r-', linewidth=2.5, label='Load')
        
        # Plot storage charging individually
        for stor_id in all_storage:
            charge = storage_charge_y10[stor_id]
            if np.any(charge < 0):  # Only plot if there's actual charging (note: charge values are negative)
                charge_color = storage_colors[stor_id]
                # Make charge color slightly different than discharge color for the same storage
                charge_color = [min(1.0, c * 0.8) for c in charge_color[:3]] + [0.3]  # Adjust color and alpha
                plt.fill_between(hours, 0, charge, color=charge_color, label=f"Storage {stor_id} (Charge)")
        
        # Plot the sum of generation (should exactly match load)
        plt.plot(hours, total_with_storage_y10, 'k--', linewidth=1.5, label='Net Generation')
        
        plt.title(f'Generation Mix for {season.capitalize()} Season - Year {last_year}', fontsize=16)
        plt.xlabel('Hour', fontsize=12)
        plt.ylabel('Power (MW)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right', fontsize=10)
        
        # Add a text box with verification results
        plt.text(0.02, 0.02, f"Maximum balance error: {max_error_y10:.6f} MW\nAverage balance error: {avg_error_y10:.6f} MW", 
                transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        
        # Save the plot
        y10_plot_file = os.path.join(output_dir, f"{season}_mix_year{last_year}.png")
        plt.savefig(y10_plot_file, dpi=300)
        plt.close()
        
        plot_files[f"{season}_year{last_year}"] = y10_plot_file
        print(f"[post.py] Created generation mix plot for {season} year {last_year} at {y10_plot_file}")
    
    return plot_files

# Comment out the generate_bus_asset_report function as it's no longer needed
'''
def generate_bus_asset_report(integrated_network, output_dir="results"):
    """
    Generate a detailed report showing the first and last 5 hours of each season,
    listing all assets (generators, storage, loads) at each bus with their actual values
    ***for the first year of the planning horizon***.
    
    Args:
        integrated_network: IntegratedNetwork object with season networks and optimization results
        output_dir: Directory to save the report
    
    Returns:
        Path to the generated report file
    """
    if not hasattr(integrated_network, 'integrated_results'):
        print("[post.py] No optimization results found")
        return None
        
    result = integrated_network.integrated_results
    if 'variables' not in result or not result['variables']:
        print("[post.py] Error: Optimization results not found or empty.")
        return None
        
    # Get the first year of the planning horizon
    if not integrated_network.years:
        print("[post.py] Error: No years defined in integrated_network.")
        return None
    first_year = integrated_network.years[0]
    
    # Get load growth factor for the first year
    load_growth_factors = getattr(integrated_network, 'load_growth', {})
    growth_factor_first_year = load_growth_factors.get(first_year, 1.0) # Default to 1.0 if not found

    # Check if variables exist
    report_file = os.path.join(output_dir, "bus_asset_report.txt")
    
    with open(report_file, 'w') as f:
        f.write(f"BUS ASSET REPORT - FIRST AND LAST 5 HOURS PER SEASON (YEAR {first_year})\n") # Indicate year
        f.write("=" * 80 + "\n\n")
        
        # For each season
        for season in integrated_network.seasons:
            f.write(f"\nSEASON: {season.upper()}\n")
            f.write("-" * 40 + "\n")
            
            network = integrated_network.season_networks[season]
            T = network.T
            
            # Limit hours shown to avoid excessive output
            hours_to_show = list(range(1, 6)) + list(range(T-4, T+1)) # First 5 and last 5 hours
            
            for hour in hours_to_show:
                f.write(f"\nHour {hour}:\n")
                
                # For each bus
                for bus_id in network.buses.index:
                    f.write(f"\n  Bus {bus_id}:\n")
                    
                    hourly_total_gen = 0
                    hourly_total_stor_dispatch = 0
                    hourly_total_load = 0

                    # List generators at this bus and sum dispatch
                    bus_gens = network.generators[network.generators['bus'] == bus_id]
                    if not bus_gens.empty:
                        f.write("    Generators:\n")
                        for gen_id in bus_gens.index:
                            gen_type = network.generators.at[gen_id, 'type']
                            gen_cap = network.generators.at[gen_id, 'p_nom']
                            
                            # Get the actual dispatch value from optimization results for the first year
                            dispatch_key = key('p_gen', season, gen_id, first_year, hour)
                            dispatch = result['variables'].get(dispatch_key, 0) # Default to 0 if key not found
                            hourly_total_gen += dispatch
                            
                            # For wind/solar, also show time-varying availability
                            # For thermal, show nominal capacity as availability
                            availability = ""
                            if gen_type in ['wind', 'solar'] and 'p_max_pu' in network.generators_t and gen_id in network.generators_t['p_max_pu']:
                                # Ensure hour is within bounds (adjust if needed, assuming hour is 1-based index)
                                if 0 <= hour - 1 < len(network.generators_t['p_max_pu'][gen_id].values):
                                     avail = network.generators_t['p_max_pu'][gen_id].values[hour-1] * gen_cap
                                     availability = f" (Available: {avail:.2f} MW)"
                                else:
                                     availability = " (Availability data missing for hour)"
                            elif gen_type == 'thermal': # <<< Add thermal availability
                                availability = f" (Available: {gen_cap:.2f} MW)" # Use nominal capacity
                            
                            f.write(f"      {gen_id} ({gen_type}): {dispatch:.2f} MW{availability}\n") # Ensure MW is present
                    
                    # List storage at this bus and sum dispatch
                    bus_storage = network.storage_units[network.storage_units['bus'] == bus_id]
                    if not bus_storage.empty:
                        f.write("    Storage:\n")
                        for stor_id in bus_storage.index:
                            # Get storage dispatch and state of charge for the first year
                            charge_key = key('p_charge', season, stor_id, first_year, hour)
                            discharge_key = key('p_discharge', season, stor_id, first_year, hour)
                            soc_key = key('soc', season, stor_id, first_year, hour)
                            
                            charge = result['variables'].get(charge_key, 0) # Default to 0 if key not found
                            discharge = result['variables'].get(discharge_key, 0) # Default to 0 if key not found
                            soc = result['variables'].get(soc_key, 0) # Default to 0 if key not found
                            
                            # Net dispatch = discharge - charge
                            net_dispatch = discharge - charge
                            hourly_total_stor_dispatch += net_dispatch
                            
                            f.write(f"      {stor_id}: Dispatch={net_dispatch:.2f} MW, SOC={soc:.2f} MWh\n")
                    
                    # List loads at this bus and sum load
                    bus_loads = network.loads[network.loads['bus'] == bus_id]
                    if not bus_loads.empty:
                        f.write("    Loads:\n")
                        for load_id in bus_loads.index:
                            # Get the actual load value from input data
                            load_value = 0
                            # Ensure hour is within bounds (adjust if needed, assuming hour is 1-based index)
                            if 'p' in network.loads_t and load_id in network.loads_t['p']:
                                 if 0 <= hour - 1 < len(network.loads_t['p'][load_id].values):
                                     load_value = network.loads_t['p'][load_id].values[hour-1]
                                 else:
                                     f.write(f"      {load_id}: Error - Load data missing for hour {hour}\n")
                                     continue # Skip this load if data missing for the hour
                            else:
                                 load_value = network.loads.at[load_id, 'p_mw'] # Use static value if no time series
                            
                            # Apply load growth factor for the first year
                            scaled_load_value = load_value * growth_factor_first_year
                            hourly_total_load += scaled_load_value
                            
                            f.write(f"      {load_id}: {scaled_load_value:.2f} MW\n")
                        f.write(f"    Total Load: {hourly_total_load:.2f} MW\n")
                    else:
                         f.write("    Loads: None\n")
                         f.write(f"    Total Load: {hourly_total_load:.2f} MW\n")
                    
                    # Calculate and write hourly net injection
                    hourly_net_injection = hourly_total_gen + hourly_total_stor_dispatch - hourly_total_load
                    f.write(f"    Net Injection: {hourly_net_injection:.2f} MW\n")
            
            f.write("\n" + "=" * 80 + "\n")
    
    print(f"[post.py] Bus asset report saved to {report_file}")
    return report_file
'''

def plot_implementation_timeline(integrated_network, output_dir="results"):
    """
    Gantt‑like view of asset life‑cycles with correct
    replacement/decommission logic.
    """
    import os, matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D

    plan = generate_implementation_plan(integrated_network, output_dir)
    if not plan:
        print("[timeline] no implementation plan – abort")
        return None

    years = integrated_network.years or plan.get("years", [])
    if not years:
        print("[timeline] missing planning years – abort")
        return None
    y0, yN = years[0], years[-1]                # inclusive horizon
    horizon_end = yN + 1                        # bar end

    season = integrated_network.seasons[0]
    net = integrated_network.season_networks[season]

    # ------------------------------------------------------------------ helpers
    def records(cat, df, mapping):
        """yield a dict per asset with timeline info."""
        recs = []
        for aid, meta in mapping.items():
            installs = sorted(set(meta["years_installed"]))
            if not installs:
                continue
            
            # Convert aid to int if it's a string, since DataFrame indices might be integers
            aid_idx = int(aid) if isinstance(aid, str) else aid
            
            # Check if aid_idx exists in the DataFrame before accessing
            if aid_idx in df.index:
                L = int(df.at[aid_idx, "lifetime_years"])
                type_label = df.at[aid_idx, 'type'] if cat == 'gen' else 'storage'
            else:
                L = 5  # Default lifetime
                type_label = 'storage' if cat != 'gen' else 'unknown'
                
            label = f"{'G' if cat=='gen' else 'S'}: {aid} ({type_label})"

            bars, repl, decomm = [], [], []
            for i, y in enumerate(installs):
                end = min(y + L, horizon_end)
                bars.append((y, end))
                # look‑ahead to classify next event
                if i + 1 < len(installs):
                    y_next = installs[i + 1]
                    if y_next == y + L:                 # seamless replacement
                        repl.append(y_next)
                    else:                               # gap → decommission then new install
                        decomm.append(end)
                else:
                    # last install → maybe final decommission
                    if end <= yN:
                        decomm.append(end)

            recs.append(dict(label=label,
                             category=cat,
                             bars=bars,
                             new=installs[0:1],         # first install only
                             repl=repl,
                             decomm=decomm))
        return recs

    rows = (records("gen",    net.generators,   plan["generators"]) +
            records("stor",   net.storage_units, plan["storage"]))

    # sort: generators first, id order
    rows.sort(key=lambda r: (0 if r["category"]=="gen" else 1, r["label"]))
    n = len(rows)
    if n == 0:
        print("[timeline] nothing to plot")
        return None

    # ------------------------------------------------------------------ plot
    h = max(6, 0.45 * n)
    fig, ax = plt.subplots(figsize=(16, h))

    y_pos = list(reversed(range(n)))
    bar_h = 0.6
    col_gen, col_sto = "#a1e8af", "#b9dff7"

    for row, y in zip(rows, y_pos):
        col = col_gen if row["category"]=="gen" else col_sto
        for s, e in row["bars"]:
            ax.add_patch(Rectangle((s, y-bar_h/2), e-s, bar_h,
                                   facecolor=col, alpha=0.6, zorder=1))

        ax.plot(row["new"],       [y]*len(row["new"]),
                "^",  mfc="green", mec="black", ms=8, zorder=3)
        ax.plot(row["repl"],      [y]*len(row["repl"]),
                "D",  mfc="orange", mec="black", ms=7, zorder=3)
        ax.plot(row["decomm"],    [y]*len(row["decomm"]),
                "x",  color="red", mew=2, ms=9, zorder=3)

    ax.set_yticks(y_pos); ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlabel("Year"); ax.set_title("Implementation Plan - Asset Timeline", fontsize=16)
    ax.set_xlim(y0-0.5, horizon_end+0.5); ax.set_ylim(-1, n)
    ax.grid(axis="x", ls="--", alpha=0.4)

    # legend
    ax.legend([
        Rectangle((0,0),1,1,fc=col_gen,  alpha=0.6),
        Rectangle((0,0),1,1,fc=col_sto,  alpha=0.6),
        Line2D([0],[0], marker="^", mfc="green", mec="black", ms=8, ls=""),
        Line2D([0],[0], marker="D", mfc="orange", mec="black", ms=7, ls=""),
        Line2D([0],[0], marker="x",  color="red", mew=2, ms=9, ls="")
    ], ["Generator Active", "Storage Active", "New Installation",
        "Replacement", "Decommission"],
       ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, "asset_timeline.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[timeline] saved {fname}")
    return fname