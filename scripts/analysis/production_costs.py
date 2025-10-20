#!/usr/bin/env python3
"""
Analysis script to calculate annual production and investment costs per asset.

This script processes the optimization results to calculate:
1. Annual production (in MWh) per generator
2. Production costs (in $) per generator
3. Investment costs (in $) per generator and storage unit
4. Total costs (production + investment) per asset
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Add parent directory to path to import from network
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules
from network import IntegratedNetwork, Network
from post import NumpyEncoder
# Import annuity calculation helpers from optimization
from optimization import compute_crf, compute_discount_sum

def calculate_production_and_costs(integrated_network):
    """
    Calculate annual production in MWh and costs per asset, including both
    production and investment costs.
    
    Args:
        integrated_network: IntegratedNetwork object with optimization results
        
    Returns:
        Dictionary with production and cost data per asset per year
    """
    if not hasattr(integrated_network, 'integrated_results') or integrated_network.integrated_results is None:
        print("No integrated_results found on integrated_network.")
        return {}

    result = integrated_network.integrated_results
    if 'variables' not in result:
        print("integrated_results has no 'variables'.")
        return {}

    var_dict = result['variables']
    years = integrated_network.years
    seasons = integrated_network.seasons
    season_weights = integrated_network.season_weights
    
    # Get a reference to the first network to extract generator and storage data
    first_network = integrated_network.season_networks[seasons[0]]
    
    # Initialize data structure to store results
    cost_data = {
        'generators': {},
        'storage': {},
        'total_annual_costs': {year: 0 for year in years},
        'total_production_mwh': {year: 0 for year in years}
    }
    
    # Process generators
    for gen_id in first_network.generators.index:
        # Get generator properties
        gen_type = first_network.generators.at[gen_id, 'type']
        gen_capex = first_network.generators.at[gen_id, 'capex']
        # Check if cost_mwh exists, otherwise try marginal_cost
        if 'cost_mwh' in first_network.generators.columns:
            gen_cost_mwh = first_network.generators.at[gen_id, 'cost_mwh']
        else:
            gen_cost_mwh = first_network.generators.at[gen_id, 'marginal_cost']
        gen_lifetime = first_network.generators.at[gen_id, 'lifetime_years']
        gen_discount_rate = first_network.generators.at[gen_id, 'discount_rate']
        # Get operating cost fraction if available, default to 0
        gen_opex_fraction = 0.0
        if 'operating_costs' in first_network.generators.columns:
            gen_opex_fraction = first_network.generators.at[gen_id, 'operating_costs']

        # Default lifetime if needed
        if gen_lifetime is None or pd.isna(gen_lifetime) or gen_lifetime <= 0:
            gen_lifetime = 1 # Avoid division by zero
            
        # Calculate the fixed annual annuity cost for this generator
        gen_discount_sum = compute_discount_sum(gen_lifetime, gen_discount_rate)
        gen_npv = gen_capex + (gen_capex * gen_opex_fraction * gen_discount_sum)
        gen_crf = compute_crf(gen_lifetime, gen_discount_rate)
        annual_annuity_gen = gen_npv * gen_crf

        # Initialize generator data
        cost_data['generators'][gen_id] = {
            'type': gen_type,
            'capex': gen_capex,
            'cost_mwh': gen_cost_mwh,
            'lifetime': gen_lifetime,
            'discount_rate': gen_discount_rate,
            'annual_production_mwh': {year: 0 for year in years},
            'annual_production_cost': {year: 0 for year in years},
            'annual_investment_cost': {year: 0 for year in years},
            'annual_total_cost': {year: 0 for year in years},
            'build_years': []
        }
        
        # Extract build decisions
        for year in years:
            build_key = f"gen_build_{gen_id}_{year}"
            if build_key in var_dict and var_dict[build_key] > 0.5:
                cost_data['generators'][gen_id]['build_years'].append(year)
                
        # Assign the fixed annual annuity cost for each year the asset is installed
        for year in years:
            installed_key = f"gen_installed_{gen_id}_{year}"
            if installed_key in var_dict and var_dict[installed_key] > 0.5:
                 cost_data['generators'][gen_id]['annual_investment_cost'][year] = annual_annuity_gen
        
        # Calculate production and production costs for each year
        for year in years:
            year_production = 0
            
            # Sum production across all seasons
            for season in seasons:
                season_weight_weeks = season_weights.get(season, 0)
                season_weight_hours = season_weight_weeks * 168  # 168 hours per week
                
                # Get the network for this season
                network = integrated_network.season_networks[season]
                T = network.T  # Number of time steps
                
                # Sum production across all hours
                for t in range(1, T + 1):
                    production_key = f"p_gen_{season}_{gen_id}_{year}_{t}"
                    if production_key in var_dict:
                        # Convert from MW to MWh by multiplying by 1 hour
                        hourly_production = var_dict[production_key]
                        # Scale by season weight to get annual equivalent
                        year_production += hourly_production * season_weight_hours / T
            
            # Store annual production in MWh
            cost_data['generators'][gen_id]['annual_production_mwh'][year] = year_production
            
            # Calculate production cost
            production_cost = year_production * gen_cost_mwh
            cost_data['generators'][gen_id]['annual_production_cost'][year] = production_cost
            
            # Calculate total cost (production + investment)
            total_cost = (production_cost + 
                          cost_data['generators'][gen_id]['annual_investment_cost'][year])
            cost_data['generators'][gen_id]['annual_total_cost'][year] = total_cost
            
            # Add to total annual costs
            cost_data['total_annual_costs'][year] += total_cost
            cost_data['total_production_mwh'][year] += year_production
    
    # Process storage units
    for stor_id in first_network.storage_units.index:
        # Get storage properties
        stor_capex = first_network.storage_units.at[stor_id, 'capex']
        stor_lifetime = first_network.storage_units.at[stor_id, 'lifetime_years']
        stor_discount_rate = first_network.storage_units.at[stor_id, 'discount_rate']
        # Get operating cost fraction if available, default to 0
        stor_opex_fraction = 0.0
        if 'operating_costs' in first_network.storage_units.columns:
            stor_opex_fraction = first_network.storage_units.at[stor_id, 'operating_costs']

        # Default lifetime if needed
        if stor_lifetime is None or pd.isna(stor_lifetime) or stor_lifetime <= 0:
            stor_lifetime = 1 # Avoid division by zero

        # Calculate the fixed annual annuity cost for this storage unit
        stor_discount_sum = compute_discount_sum(stor_lifetime, stor_discount_rate)
        stor_npv = stor_capex + (stor_capex * stor_opex_fraction * stor_discount_sum)
        stor_crf = compute_crf(stor_lifetime, stor_discount_rate)
        annual_annuity_stor = stor_npv * stor_crf

        # Initialize storage data
        cost_data['storage'][stor_id] = {
            'capex': stor_capex,
            'lifetime': stor_lifetime,
            'discount_rate': stor_discount_rate,
            'annual_investment_cost': {year: 0 for year in years},
            'annual_total_cost': {year: 0 for year in years},
            'build_years': []
        }
        
        # Extract build decisions
        for year in years:
            build_key = f"storage_build_{stor_id}_{year}"
            if build_key in var_dict and var_dict[build_key] > 0.5:
                cost_data['storage'][stor_id]['build_years'].append(year)
        
        # Assign the fixed annual annuity cost for each year the asset is installed
        for year in years:
            installed_key = f"storage_installed_{stor_id}_{year}"
            if installed_key in var_dict and var_dict[installed_key] > 0.5:
                 cost_data['storage'][stor_id]['annual_investment_cost'][year] = annual_annuity_stor
                 cost_data['storage'][stor_id]['annual_total_cost'][year] = annual_annuity_stor # Storage only has investment cost here
                 
                 # Add to total annual costs
                 # Note: We should sum total costs AFTER calculating them per asset, not here.
                 # This is handled later in the generator loop. We need to adjust total cost summing.

    # Correctly calculate total annual costs after processing all assets
    for year in years:
        total_year_cost = 0
        # Sum generator costs
        for gen_id, gen_data in cost_data['generators'].items():
            total_year_cost += gen_data['annual_total_cost'][year]
        # Sum storage costs (which are just investment costs in this setup)
        for stor_id, stor_data in cost_data['storage'].items():
             total_year_cost += stor_data['annual_total_cost'][year] # already set to annuity if installed
        cost_data['total_annual_costs'][year] = total_year_cost

    return cost_data

def print_cost_report(cost_data):
    """
    Print a detailed report of production and costs.
    
    Args:
        cost_data: Dictionary with production and cost data per asset per year
    """
    print("\n" + "="*80)
    print("PRODUCTION AND COST ANALYSIS REPORT")
    print("="*80)
    
    # Print generator data
    print("\nGENERATOR PRODUCTION AND COSTS:")
    print("-"*80)
    
    for gen_id, gen_data in cost_data['generators'].items():
        print(f"\nGenerator {gen_id} ({gen_data['type']}):")
        print(f"  Capex: ${gen_data['capex']:,.2f}")
        print(f"  Marginal Cost: ${gen_data['cost_mwh']:,.2f}/MWh")
        print(f"  Lifetime: {gen_data['lifetime']} years")
        print(f"  Build years: {gen_data['build_years']}")
        
        print("\n  Year | Production (MWh) | Production Cost ($) | Investment Cost ($) | Total Cost ($)")
        print("  " + "-"*75)
        
        for year in sorted(gen_data['annual_production_mwh'].keys()):
            production = gen_data['annual_production_mwh'][year]
            prod_cost = gen_data['annual_production_cost'][year]
            inv_cost = gen_data['annual_investment_cost'][year]
            total_cost = gen_data['annual_total_cost'][year]
            
            print(f"  {year:4d} | {production:15,.2f} | {prod_cost:18,.2f} | {inv_cost:18,.2f} | {total_cost:14,.2f}")
    
    # Print storage data
    print("\n\nSTORAGE COSTS:")
    print("-"*80)
    
    for stor_id, stor_data in cost_data['storage'].items():
        print(f"\nStorage {stor_id}:")
        print(f"  Capex: ${stor_data['capex']:,.2f}")
        print(f"  Lifetime: {stor_data['lifetime']} years")
        print(f"  Build years: {stor_data['build_years']}")
        
        print("\n  Year | Investment Cost ($) | Total Cost ($)")
        print("  " + "-"*50)
        
        for year in sorted(stor_data['annual_investment_cost'].keys()):
            inv_cost = stor_data['annual_investment_cost'][year]
            total_cost = stor_data['annual_total_cost'][year]
            
            print(f"  {year:4d} | {inv_cost:18,.2f} | {total_cost:14,.2f}")
    
    # Print system summary
    print("\n\nSYSTEM SUMMARY:")
    print("-"*80)
    print("\n  Year | Total Production (MWh) | Total System Cost ($)")
    print("  " + "-"*50)
    
    for year in sorted(cost_data['total_annual_costs'].keys()):
        total_prod = cost_data['total_production_mwh'][year]
        total_cost = cost_data['total_annual_costs'][year]
        
        print(f"  {year:4d} | {total_prod:22,.2f} | {total_cost:21,.2f}")
    
    print("\n" + "="*80)

def save_cost_data(cost_data, output_dir="results"):
    """
    Save the cost data to a JSON file.
    
    Args:
        cost_data: Dictionary with production and cost data per asset per year
        output_dir: Directory to save the output file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "production_costs.json")
    
    with open(output_file, "w") as f:
        json.dump(cost_data, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nCost data saved to {output_file}")
    return output_file

def plot_production_costs(cost_data, output_dir="results"):
    """
    Create visualizations of production and costs.
    
    Args:
        cost_data: Dictionary with production and cost data per asset per year
        output_dir: Directory to save the output plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Total annual production by generator ID (instead of type)
    years = sorted(list(cost_data['total_annual_costs'].keys()))
    
    # Calculate production by generator ID and year
    production_by_gen = {}
    for gen_id, gen_data in cost_data['generators'].items():
        production_by_gen[gen_id] = {year: 0 for year in years}
        for year in years:
            production_by_gen[gen_id][year] = cost_data['generators'][gen_id]['annual_production_mwh'][year]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Stack the production
    bottom = np.zeros(len(years))
    for gen_id in production_by_gen:
        values = [production_by_gen[gen_id][year] for year in years]
        plt.bar(years, values, label=f"Gen {gen_id}", bottom=bottom)
        bottom += np.array(values)
    
    plt.title('Annual Electricity Production by Generator', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Production (MWh)', fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot
    production_plot = os.path.join(output_dir, "annual_production_by_generator.png")
    plt.savefig(production_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Annual costs by category (production vs. investment)
    production_costs = {year: 0 for year in years}
    investment_costs = {year: 0 for year in years}
    
    # Sum up all production costs from generators
    for gen_id, gen_data in cost_data['generators'].items():
        for year in years:
            production_costs[year] += gen_data['annual_production_cost'][year]
            investment_costs[year] += gen_data['annual_investment_cost'][year]
    
    # Add storage investment costs
    for stor_id, stor_data in cost_data['storage'].items():
        for year in years:
            investment_costs[year] += stor_data['annual_investment_cost'][year]
    
    # Create the stacked bar chart
    plt.figure(figsize=(12, 6))
    
    prod_values = [production_costs[year] for year in years]
    inv_values = [investment_costs[year] for year in years]
    
    plt.bar(years, prod_values, label='Production Costs', color='blue')
    plt.bar(years, inv_values, bottom=prod_values, label='Investment Costs', color='orange')
    
    plt.title('Annual System Costs by Category', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Cost ($)', fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot
    cost_plot = os.path.join(output_dir, "annual_costs_by_category.png")
    plt.savefig(cost_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'production_plot': production_plot,
        'cost_plot': cost_plot
    }

def analyze_production_costs(integrated_network, output_dir="results"):
    """
    Main function to analyze production and costs.
    
    Args:
        integrated_network: IntegratedNetwork object with optimization results
        output_dir: Directory to save the output files
    """
    print("Analyzing production and costs...")
    
    # Calculate production and costs
    cost_data = calculate_production_and_costs(integrated_network)
    
    # Print the report
    print_cost_report(cost_data)
    
    # Save the data to a JSON file
    save_cost_data(cost_data, output_dir)
    
    # Create and save plots
    plot_production_costs(cost_data, output_dir)
    
    print("Production and cost analysis completed.")
    
    return cost_data

# If called directly
if __name__ == "__main__":
    print("This script should be imported and used as a module.")
    print("Example usage:")
    print("  from scripts.analysis.production_costs import analyze_production_costs")
    print("  analyze_production_costs(integrated_network)") 