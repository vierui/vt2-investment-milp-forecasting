#!/usr/bin/env python3
"""
Component classes for power grid network modeling.

These are simple data classes or minimal Python classes that store
data about Buses, Generators, Loads, Storage, and Branches.
"""

import pandas as pd

class Bus:
    """Represents a node in the power network."""
    def __init__(self, index, name, v_nom=1.0):
        self.index = index
        self.name = name
        self.v_nom = v_nom

class Generator:
    """Represents a power generation unit."""
    def __init__(self, index, name, bus, p_nom=0, marginal_cost=0,
                 gen_type='thermal', capex_per_mw=0, lifetime_years=20):
        self.index = index
        self.name = name
        self.bus = bus
        self.p_nom = p_nom
        self.marginal_cost = marginal_cost
        self.type = gen_type
        self.capex_per_mw = capex_per_mw
        self.lifetime_years = lifetime_years

class Load:
    """Represents a power load/demand."""
    def __init__(self, index, name, bus, p_mw=0):
        self.index = index
        self.name = name
        self.bus = bus
        self.p_mw = p_mw

class Storage:
    """Represents an energy storage unit."""
    def __init__(self, index, name, bus, p_nom=0,
                 efficiency_store=0.9, efficiency_dispatch=0.9,
                 max_hours=6, capex_per_mw=0, lifetime_years=10):
        self.index = index
        self.name = name
        self.bus = bus
        self.p_nom = p_nom
        self.efficiency_store = efficiency_store
        self.efficiency_dispatch = efficiency_dispatch
        self.max_hours = max_hours
        self.capex_per_mw = capex_per_mw
        self.lifetime_years = lifetime_years

class Branch:
    """Represents a transmission line (branch)."""
    def __init__(self, index, name, from_bus, to_bus, susceptance, s_nom=0):
        self.index = index
        self.name = name
        self.from_bus = from_bus
        self.to_bus = to_bus
        self.susceptance = susceptance
        self.s_nom = s_nom