
# Investment model for energy assets : Power-flow MILP optimization and lifetime-cycle planning

## OVERVIEW
VT2 is an optimization and modeling toolkit for long-term energy infrastructure planning. It helps planners and engineers determine what to build, when to build it, and how to operate it over a multi-decade horizon, using a robust mixed-integer linear programming (MILP) core and integrated data-driven forecasting.

This project is the direct continuation of VT1, expanding from single-year and scenario-based analysis to detailed multi-period, lifecycle-aware decision making.

## MOTIVATION
Energy system investment planning is complex: planners must decide what to build, when to build it, and how to operate it as costs, technologies, and demand evolve. Traditional scenario-based methods are limited—they require manual case definition, cannot capture asset lifecycles, and often miss better solutions hidden in the combinatorial space.

This milp-investment model addresses these challenges by:
- Replacing scenario-by-scenario analysis with direct global optimization over all possible asset mixes and schedules.
- Explicitly tracking asset lifecycles and retirement.
- Allowing the model to automatically select and dispatch the most cost-effective assets from the full candidate pool.
- Efficiently solving large, multi-year planning problems with vectorized constraint formulation.
- Supporting dynamic system needs with integrated load growth and renewable forecasting modules.

## FEATURES
### MILP Optimization
- Multi-year investment planning: Optimizes asset commissioning, operation, and retirement across the full planning horizon.
- Automatic asset selection: Chooses the optimal mix and timing from all available candidates.
- Vectorized constraint formulation: Efficiently builds large optimization problems for scalable performance.
- Annualized cost modeling: Accurately reflects capital and operating costs year by year.
- Dynamic demand adaptation: Updates asset decisions as projected load evolves.
- Asset lifecycle management: Tracks asset status for installation, operation, and retirement within the model.

### Forecasting
- Integrated PV forecasting module: Uses machine learning (XGBoost + weather data) for short-term solar generation predictions.
- Feature engineering: Incorporates time, lag, and weather-based predictors for improved forecast accuracy.
- Scenario-ready outputs: Enables robust planning by quantifying renewable uncertainty and its operational impacts.

## GETTING STARTED
### Requirements
- **Python**: 3.10 or later
- **Package Manager**: Poetry (pip install poetry)
- **Solver**: IBM CPLEX Studio 22.1+ ([IBM Academic License](https://www.ibm.com/academic))

*Note*: CPLEX path in pyproject.toml must match your installation location

### Install
```
git clone https://github.com/vierui/investment-milp-forecasting.git
cd investment-milp-forecasting
poetry install
poetry shell
```
### Run a sample scenario
```
python scripts/main.py --grid-file data/grid --profiles-dir data/processed --output-dir results/milp
```
_Default: 30-year horizon, M4 MacBook Pro < 10 min_

### Run PV forecasting (optional)
```
python notebooks/E-weatherfeatures.py
```

## EXAMPLE OUTPUTS AND RESULTS
- Asset Timeline

Shows when assets are built, operated, and retired.
- Generation Mix (Years 5 vs. 25)

Illustrates portfolio evolution as demand grows.
- Forecasting Results

Compares XGBoost vs. SARIMA for PV prediction.

## HOW IT WORKS 
- Inputs:
   - Grid and asset data (CSV/json)
   - Load growth, seasonal profiles (processed via clustering)
- Optimization:
   - Formulated in vectorized MILP (CPLEX via CVXPY)
   - Binary variables track builds and operational status per asset, per year
   - Cost function annualized for fair asset comparison
- Lifecycle Logic:
   - Assets installed for finite lifetimes; model handles retirement/replacement
   - Demand evolution triggers re-optimization of the mix and timing
- Forecasting:
   - PV generation pipeline (optional) uses time, lagged, and weather features; supports rapid model experimentation

## FUTUR WORK 
- Add constraints in dcopf algorithm (e.g. ramp-up /-down)
- Experiment with alternative forecasting models or physical features 
- Robust model testing for ML forecasting
- Scale to larger grids—see results section for benchmarks

## ACKNOWLEDGEMENTS
Built as part of a master’s project at ZHAW School of Engineering.
Thanks to the open-source Python community, ZHAW IEFE for support, and all contributors to CVXPY, XGBoost, and pvlib.

For technical details and methods, see the project report in report/main.pdf. Questions or feedback welcome. Feel free to reach out or open an issue!

**License**: Academic use only. Please cite appropriately if this work supports your research.