#!/usr/bin/env python3
"""
Network module for the simplified multi-year approach.

Classes:
  - IntegratedNetwork: A container for multiple (season) sub-networks 
                       and a list of planning years.
  - Network: A single-season or single-scenario network object with:
      buses, lines, generators, loads, storage_units, time-series info.

No references to re-install, replacement, or complicated multi-binary logic 
inside these classes. They simply store the data. The new optimization is 
handled in optimization.py.
"""

import os
import pickle
import pandas as pd
import numpy as np


class IntegratedNetwork:
    """
    Container for multiple seasonal sub-networks and multi-year planning data.
    
    The new simplified approach:
      - 'seasons' is a list of names, e.g. ['winter','summer','spri_autu'].
      - 'years' is a list of relative year indices, e.g. [1,2,3,...].
      - 'discount_rate' can be stored if needed, but we may ignore 
        if not applying discounting.
      - 'season_networks' is a dict {season_name: Network}.
      - 'season_weights' can hold e.g. {'winter':13, 'summer':13, 'spri_autu':26}.
      
    This class does NOT store replacements or multiple binary variables. 
    It is purely a container. The actual optimization is in optimization.py.
    """

    def __init__(self, seasons=None, years=None, discount_rate=0.0, season_weights=None):
        """
        Args:
            seasons (list): e.g. ['winter','summer','spri_autu']
            years (list): e.g. [1,2,3,...]
            discount_rate (float): optional discount rate
            season_weights (dict): optional map from season->weeks
        """
        self.seasons = seasons or []
        self.years = years or []
        self.discount_rate = discount_rate
        
        self.season_networks = {}  # dict: season_name -> Network
        self.season_weights = season_weights or {
            'winter': 13,
            'summer': 13,
            'spri_autu': 26
        }

        # Optionally store final integrated results from optimization
        self.integrated_results = None

    def add_season_network(self, season, network):
        """
        Adds a Network object for a particular season to this integrated structure.
        If the season is not in self.seasons, we append it.
        """
        if season not in self.seasons:
            self.seasons.append(season)
        self.season_networks[season] = network

    def get_season_network(self, season):
        """
        Returns the stored sub-network for a given season name.
        """
        return self.season_networks.get(season)

    def save_to_pickle(self, filename):
        """
        Save the entire IntegratedNetwork to a pickle file.
        """
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
            print(f"[network.py] IntegratedNetwork saved to {filename}")
            return True
        except Exception as e:
            print(f"[network.py] Error saving IntegratedNetwork: {e}")
            return False

    @classmethod
    def load_from_pickle(cls, filename):
        """
        Load an IntegratedNetwork from a pickle file.
        """
        if not os.path.exists(filename):
            print(f"[network.py] File not found: {filename}")
            return None
        try:
            with open(filename, 'rb') as f:
                net = pickle.load(f)
            print(f"[network.py] IntegratedNetwork loaded from {filename}")
            return net
        except Exception as e:
            print(f"[network.py] Error loading IntegratedNetwork: {e}")
            return None


class Network:
    """
    Represents a single-season or single-scenario network.
    Stores data frames for buses, lines, generators, loads, storage_units,
    plus time steps (snapshots), etc.
    
    This is intentionally minimal. The actual power flow or 
    multi-year constraints happen in optimization.py. 
    """

    def __init__(self, name=""):
        self.name = name

        # DataFrames for each component
        self.buses = pd.DataFrame(columns=['name','v_nom'])
        self.lines = pd.DataFrame(columns=['name','from_bus','to_bus','susceptance','s_nom'])
        self.generators = pd.DataFrame(columns=['name','bus','p_nom','marginal_cost',
                                                'type','capex','lifetime_years','discount_rate'])
        self.storage_units = pd.DataFrame(columns=['name','bus','p_nom',
                                                   'efficiency_store','efficiency_dispatch','max_hours',
                                                   'capex','lifetime_years','discount_rate'])
        self.loads = pd.DataFrame(columns=['name','bus','p_mw'])

        # Timeseries
        self.snapshots = pd.Index([])
        self.T = 0  # number of time steps in this network
        # optional placeholders for time-series data
        self.loads_t = {'p': {}}
        self.generators_t = {'p': {}, 'p_max_pu': {}}
        self.storage_units_t = {'p_charge': {}, 'p_discharge': {}, 'soc': {}}

    def create_snapshots(self, start_time, periods, freq='h'):
        """
        Creates a date_range for the specified number of periods from start_time.
        T is set to number of snapshots.
        """
        self.snapshots = pd.date_range(start=start_time, periods=periods, freq=freq)
        self.T = len(self.snapshots)
        return self.snapshots

    def add_load_time_series(self, load_id, p_values):
        """
        Example method to store time-dependent load for a given load_id.
        """
        if len(p_values) != self.T:
            print(f"[network.py] Mismatch in length of load timeseries (got {len(p_values)}, expected {self.T})")
            return False

        # Store the time series
        self.loads_t['p'][load_id] = pd.Series(p_values, index=self.snapshots)
        
        # Log some statistics about the load profile
        print(f"[network.py] Load {load_id} time series added:")
        print(f"  - Length: {len(p_values)} hours")
        print(f"  - Min load: {min(p_values):.2f} MW")
        print(f"  - Max load: {max(p_values):.2f} MW")
        print(f"  - Average load: {sum(p_values)/len(p_values):.2f} MW")
        
        return True

    def add_generator_time_series(self, gen_id, p_max_values):
        """
        Store time-dependent maximum output profile for a generator.
        
        Args:
            gen_id: Generator ID to add the time series for
            p_max_values: Array of maximum output values (in MW or per-unit)
            
        Returns:
            Boolean indicating success
        """
        if len(p_max_values) != self.T:
            print(f"[network.py] Mismatch in length of generator timeseries (got {len(p_max_values)}, expected {self.T})")
            return False
            
        # Store the time series in the generators_t dictionary
        self.generators_t['p_max_pu'][gen_id] = pd.Series(p_max_values, index=self.snapshots)
        
        # Log some statistics about the generator profile
        print(f"[network.py] Generator {gen_id} time series added:")
        print(f"  - Length: {len(p_max_values)} hours")
        print(f"  - Min availability: {min(p_max_values):.2f}")
        print(f"  - Max availability: {max(p_max_values):.2f}")
        print(f"  - Average availability: {sum(p_max_values)/len(p_max_values):.2f}")
        
        return True

    def save_to_pickle(self, filepath):
        """
        Saves this single Network to a pickle. 
        (Often you'd use IntegratedNetwork's save, but this is here for completeness.)
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            print(f"[network.py] Network '{self.name}' saved to {filepath}")
            return True
        except Exception as e:
            print(f"[network.py] Error saving Network '{self.name}': {e}")
            return False

    @staticmethod
    def load_from_pickle(filepath):
        """
        Loads a single Network from a pickle file.
        """
        if not os.path.exists(filepath):
            print(f"[network.py] Pickle not found: {filepath}")
            return None
        try:
            with open(filepath, 'rb') as f:
                net = pickle.load(f)
            print(f"[network.py] Network loaded from {filepath}")
            return net
        except Exception as e:
            print(f"[network.py] Error loading Network: {e}")
            return None