#!/usr/bin/env python3
"""
Simplified pre-processing module for multi-year, 3 representative weeks approach.
Returns a dictionary with:
  - 'grid_data': { 'buses', 'lines', 'generators', 'loads', 'storage_units', 'analysis' }
  - 'seasons_profiles': { 'winter': { 'loads', 'generators', 'hours', ...}, ...}
  
We assume each season profile includes 'hours' (e.g. 168 for a single week),
and possibly a 'loads' time series factor or similar.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# You can tweak these or load them from "analysis.json" if you prefer
SEASON_WEEKS = {
    'winter': 2,      # e.g. pick 2nd week of year
    'summer': 32,     # e.g. pick 32nd week for "summer"
    'spri_autu': 43   # a sample for "spring/autumn"
}

# How many weeks each season "represents" - Now moved to analysis.json

def get_week_start(year, week_number):
    """
    Return the Monday of the given (year, week_number).
    """
    # Monday of the first week is basically: Jan 1 + (week_number-1)*7 days,
    # but we ensure we get the exact Monday using weekday adjustments:
    jan1 = datetime(year, 1, 1)
    offset_days = (week_number - 1) * 7
    dt = jan1 + timedelta(days=offset_days)
    # shift to Monday
    monday_dt = dt - timedelta(days=dt.weekday())  
    return monday_dt

def load_csv_if_exists(path, **kwargs):
    if not os.path.exists(path):
        print(f"[pre.py] Warning: File not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, **kwargs)
    print(f"[pre.py] Loaded {path} with columns: {df.columns.tolist()}")
    return df

def load_grid_data(grid_dir):
    """
    Loads the CSV files for buses, lines, generators, loads, storages, plus analysis.json.
    Returns a dict with DataFrames.
    """
    grid_data = {}

    # Buses
    buses = load_csv_if_exists(os.path.join(grid_dir, "buses.csv"))
    if not buses.empty:
        grid_data['buses'] = buses

    # Lines
    lines = load_csv_if_exists(os.path.join(grid_dir, "lines.csv"))
    if not lines.empty:
        grid_data['lines'] = lines

    # Generators
    gens = load_csv_if_exists(os.path.join(grid_dir, "generators.csv"))
    if not gens.empty:
        grid_data['generators'] = gens

    # Loads
    loads = load_csv_if_exists(os.path.join(grid_dir, "loads.csv"))
    if not loads.empty:
        grid_data['loads'] = loads

    # Storage
    stor_csv = os.path.join(grid_dir, "storage_units.csv")
    if not os.path.exists(stor_csv):
        stor_csv = os.path.join(grid_dir, "storages.csv")
    storages = load_csv_if_exists(stor_csv)
    if not storages.empty:
        grid_data['storage_units'] = storages

    # Analysis (planning horizon) from analysis.json
    analysis_file = os.path.join(grid_dir, "analysis.json")
    analysis = {}
    if os.path.exists(analysis_file):
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)
    grid_data['analysis'] = analysis
    
    # pull season weights here so the caller doesn't have to reopen the file
    grid_data['season_weights'] = analysis.get("representative_weeks", {})

    return grid_data

def load_processed_time_series(profiles_dir):
    """
    Loads any raw/processed time series data you have (e.g. load-2023.csv, wind-2023.csv, etc.).
    Returns them in a dict.
    """
    data = {}
    # Example: load
    load_path = os.path.join(profiles_dir, "load-2023.csv")
    if os.path.exists(load_path):
        df = pd.read_csv(load_path, parse_dates=['time'])
        data['load'] = df
    # similarly for wind, solar...
    wind_path = os.path.join(profiles_dir, "wind-2023.csv")
    if os.path.exists(wind_path):
        df = pd.read_csv(wind_path, parse_dates=['time'])
        data['wind'] = df

    solar_path = os.path.join(profiles_dir, "solar-2023.csv")
    if os.path.exists(solar_path):
        df = pd.read_csv(solar_path, parse_dates=['time'])
        data['solar'] = df

    return data

def extract_week_data(raw_df, week_start, week_hours=168):
    """
    Given a DataFrame with a 'time' column, slice out the 168-hour block starting from week_start.
    Return just that slice as a DataFrame.
    """
    if 'time' not in raw_df.columns:
        return raw_df.head(0)  # empty

    mask = (raw_df['time'] >= week_start) & (raw_df['time'] < week_start + timedelta(hours=week_hours))
    return raw_df[mask].copy()

def prepare_generator_profiles(season_data, generators_df):
    """
    Prepare generator profiles for generators based on their type.
    This function creates a DataFrame with generator availability profiles.
    
    Args:
        season_data: Dictionary with wind_df and solar_df from the season
        generators_df: DataFrame with generator information (must have 'type' column)
        
    Returns:
        DataFrame with multi-index (time, gen_id) containing availability profiles
    """
    # Start with an empty DataFrame
    gen_profiles = []
    
    # If we have wind data and there are wind generators
    if 'wind_df' in season_data and not season_data['wind_df'].empty:
        wind_gens = generators_df[generators_df['type'] == 'wind']
        for _, gen in wind_gens.iterrows():
            gen_id = gen['id']
            # For each time step in the wind data
            for i, row in season_data['wind_df'].iterrows():
                time = row['time']
                value = row['value']  # Assuming this is in MW
                gen_profiles.append({
                    'time': time,
                    'gen_id': gen_id,
                    'p_max_pu': value  # Pass the raw value
                })
    
    # If we have solar data and there are solar generators
    if 'solar_df' in season_data and not season_data['solar_df'].empty:
        solar_gens = generators_df[generators_df['type'] == 'solar']
        for _, gen in solar_gens.iterrows():
            gen_id = gen['id']
            # For each time step in the solar data
            for i, row in season_data['solar_df'].iterrows():
                time = row['time']
                value = row['value']  # Assuming this is in MW
                gen_profiles.append({
                    'time': time,
                    'gen_id': gen_id,
                    'p_max_pu': value  # Pass the raw value
                })
    
    # Convert to DataFrame and set index
    if gen_profiles:
        profiles_df = pd.DataFrame(gen_profiles)
        profiles_df.set_index(['time', 'gen_id'], inplace=True)
        return profiles_df
    else:
        # Create an empty DataFrame with the correct structure
        # This avoids the KeyError when setting index on empty DataFrame
        df = pd.DataFrame({'time': [], 'gen_id': [], 'p_max_pu': []})
        return df.set_index(['time', 'gen_id'])

def prepare_load_profiles(season_data, loads_df):
    """
    Prepare load profiles from load data. This function creates a DataFrame 
    with time-varying load profiles.
    
    Args:
        season_data: Dictionary with load_df from the season
        loads_df: DataFrame with load information
        
    Returns:
        DataFrame with multi-index (time, load_id) containing load profiles
    """
    # Start with an empty list
    load_profiles = []
    
    # If we have load data
    if 'load_df' in season_data and not season_data['load_df'].empty:
        for _, load in loads_df.iterrows():
            load_id = load['id']
            # For each time step in the load data
            for i, row in season_data['load_df'].iterrows():
                time = row['time']
                value = row['value']  # This is the time-varying factor
                
                # Add entry for this load at this time
                load_profiles.append({
                    'time': time,
                    'load_id': load_id,
                    'p_pu': value  # Store the time-varying factor
                })
    
    # Convert to DataFrame and set index
    if load_profiles:
        profiles_df = pd.DataFrame(load_profiles)
        profiles_df.set_index(['time', 'load_id'], inplace=True)
        return profiles_df
    else:
        # Create an empty DataFrame with the correct structure
        # This avoids the KeyError when setting index on empty DataFrame
        df = pd.DataFrame({'time': [], 'load_id': [], 'p_pu': []})
        return df.set_index(['time', 'load_id'])

def process_data_for_optimization(grid_dir, processed_dir, planning_years=None):
    """
    Main entry point for pre-processing. Loads grid data & time series,
    extracts 3 representative weeks, returns a dict with:
      - 'grid_data': {DataFrames} 
      - 'seasons_profiles': { 'winter': {...}, 'summer': {...}, 'spri_autu': {...} }

    If you have multi-year expansions, you typically still reuse the same 3 weeks,
    possibly scaling loads in the optimization. This script won't do the scaling 
    because we do that in the model or the 'Network' objects themselves.
    """

    # 1) Load all CSV-based grid data
    grid_data = load_grid_data(grid_dir)

    # 2) Load time series from processed_dir
    raw_data = load_processed_time_series(processed_dir)

    # If there's analysis data with "planning_horizon", we can read years
    if 'analysis' in grid_data and 'planning_horizon' in grid_data['analysis']:
        ph = grid_data['analysis']['planning_horizon']
        if planning_years is not None:
            # override if the user provided a CLI arg
            ph['years'] = list(range(1, planning_years + 1))
        # else keep them as is if they exist
    else:
        # create a fallback
        grid_data.setdefault('analysis', {})
        grid_data['analysis'].setdefault('planning_horizon', {})
        grid_data['analysis']['planning_horizon']['years'] = [1]

    # 3) Build the 'seasons_profiles'
    # We'll store for each of 'winter','summer','spri_autu': 
    #   a dictionary with 'hours', plus optional 'loads', 'generators' etc.
    seasons_profiles = {}
    base_year = 2023  # or get from analysis

    for season, w_num in SEASON_WEEKS.items():
        # find Monday for that representative week
        start_dt = get_week_start(base_year, w_num)
        # slice load, wind, solar for that 168h block
        # store them for future usage
        block_hours = 168  # standard 1 week
        sub_data = {}

        # Extract load data for this season
        if 'load' in raw_data:
            extracted_load = extract_week_data(raw_data['load'], start_dt, block_hours)
            sub_data['load_df'] = extracted_load

        # Extract wind and solar data for this season
        if 'wind' in raw_data:
            wind_slice = extract_week_data(raw_data['wind'], start_dt, block_hours)
            sub_data['wind_df'] = wind_slice

        if 'solar' in raw_data:
            solar_slice = extract_week_data(raw_data['solar'], start_dt, block_hours)
            sub_data['solar_df'] = solar_slice

        # We'll store how many hours we have
        # If it doesn't have 168, we can clamp or fill
        actual_hours = block_hours
        sub_data['hours'] = block_hours

        # Create load profiles
        if 'loads' in grid_data:
            loads_df = grid_data['loads']
            load_profiles = prepare_load_profiles(sub_data, loads_df)
            if not load_profiles.empty:
                sub_data['loads'] = load_profiles

        # Create generator profiles based on type (wind, solar)
        if 'generators' in grid_data:
            generators_df = grid_data['generators']
            generator_profiles = prepare_generator_profiles(sub_data, generators_df)
            if not generator_profiles.empty:
                sub_data['generators'] = generator_profiles

        seasons_profiles[season] = sub_data

    # ---- season‑weight sanity check ----
    weights = grid_data.get('season_weights', {})
    if not weights or abs(sum(weights.values())-52) > 1e-6:
        print("[pre.py] Warning: representative_weeks missing or not summing to 52; using default 13/13/26.")
        weights = {'winter':13, 'summer':13, 'spri_autu':26}

    return {
        'grid_data': grid_data,
        'seasons_profiles': seasons_profiles,
        'season_weights': weights
    }