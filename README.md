# 🏎️ F1 2026 Championship Prediction Using Machine Learning

An end-to-end machine learning project that estimates **Formula 1 2026 World Drivers' Championship probabilities** using historical F1 data, driver and constructor performance features, Random Forest regression, Monte Carlo simulation, historical backtesting, and probability calibration.

The system predicts driver finishing positions for upcoming races and uses thousands of simulated championship outcomes to estimate the probability of each driver winning the 2026 Drivers' Championship.

---

## 🚀 Project Overview

Predicting a Formula 1 championship is different from predicting the winner of a single race.

Instead of directly training a model to predict the championship winner, this project uses a **race-by-race prediction approach**.

The model first estimates a driver's finishing position in an upcoming race. These predictions are then used inside a Monte Carlo simulation to generate thousands of possible championship outcomes.

The resulting probabilities are then calibrated using historical F1 seasons.

### Overall Pipeline

```text
Historical F1 Data
        │
        ▼
Data Collection & Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Time-Based Model Training
        │
        ▼
Random Forest Regression
        │
        ▼
Race Finishing Position Prediction
        │
        ▼
Monte Carlo Simulation
        │
        ▼
Historical Backtesting
        │
        ▼
Probability Calibration
        │
        ▼
2026 Championship Probabilities
        │
        ▼
Streamlit Dashboard
🎯 Objectives

The project was designed to:

Predict driver finishing positions for upcoming races.
Use historical driver and constructor performance.
Incorporate championship standings into race predictions.
Avoid future-data leakage through chronological feature engineering.
Evaluate the model using time-based validation.
Simulate thousands of possible championship outcomes.
Estimate championship-winning probabilities.
Backtest the simulation against previous F1 seasons.
Calibrate championship probabilities using historical out-of-sample results.
Present the results through an interactive Streamlit dashboard.
Create a framework that can be updated after each completed race.
📊 Dataset

Historical Formula 1 data from 2018–2025 is used for model development and evaluation.

The project uses several categories of F1 data:

Race Results

Includes information such as:

Driver
Constructor
Race
Round
Finishing position
Points
Status
Laps
Grid position
Qualifying Results

Used to capture qualifying performance and starting-position-related information.

Sprint Results

Sprint results are incorporated into the championship simulation and historical data pipeline.

Driver Standings

Used to calculate the driver's championship state before each race.

Constructor Standings

Used to capture constructor championship performance before each race.

🧠 Feature Engineering

The model uses driver-level, constructor-level, and championship-level features.

All historical rolling features are constructed using information available before the target race.

This helps reduce the risk of information leakage.

Driver Performance Features

The model includes rolling driver performance metrics such as:

driver_avg_finish_last_3
driver_avg_finish_last_5

driver_avg_points_last_3
driver_avg_points_last_5

driver_avg_qualifying_last_3
driver_avg_qualifying_last_5

driver_dnf_rate

These features capture recent driver performance rather than relying only on season-level statistics.

Constructor Performance Features

Constructor performance is also incorporated:

constructor_avg_finish_last_3
constructor_avg_points_last_3
constructor_avg_qualifying_last_3

This is important because F1 performance is strongly influenced by the competitiveness of the car.

Championship State Features

The model also uses the championship situation immediately before the target race:

driver_championship_position_before
driver_championship_points_before
driver_wins_before

constructor_championship_position_before
constructor_championship_points_before
constructor_wins_before

These features provide the model with information about championship momentum and current standings.

🤖 Machine Learning Model

The primary model is a:

Random Forest Regressor

The model predicts the expected finishing position of a driver rather than directly predicting the championship winner.

Model Configuration
n_estimators = 300
max_depth = 12
min_samples_leaf = 3
random_state = 42
n_jobs = -1

Categorical features such as driver and constructor are encoded using one-hot encoding.

Numerical features use median imputation where necessary.

🔄 Why Predict Race Finishing Position?

Directly predicting the championship winner would provide very few training examples because each season has only one champion.

Instead, the project decomposes the problem:

Predict Race Performance
        ↓
Convert Finishing Positions → Points
        ↓
Simulate Remaining Races
        ↓
Calculate Final Championship Standings
        ↓
Estimate Championship Probabilities

This provides the model with thousands of driver-race observations instead of only a handful of championship outcomes.

🧪 Time-Based Model Validation

A random train/test split was avoided because Formula 1 data is chronological.

Using future races to predict past races could introduce unrealistic information leakage.

The main evaluation strategy was:

Training
2018–2023
    │
    ▼
Validation
2024
    │
    ▼
Testing
2025
Pre-Qualifying Model Performance
Dataset	MAE	RMSE
Training	2.374	3.040
Validation — 2024	3.485	4.364
Test — 2025	3.853	4.758

The 2025 season was treated as an unseen test season.

🎲 Monte Carlo Championship Simulation

The race prediction model provides expected finishing positions.

However, a single deterministic prediction does not represent the uncertainty involved in Formula 1.

To account for this uncertainty, the project uses Monte Carlo simulation.

Simulation Process

For every simulation:

Start from the current championship standings.
Predict driver performance for remaining races.
Add uncertainty around the predicted finishing positions.
Account for driver DNF probability.
Convert simulated finishing positions into championship points.
Simulate remaining races.
Calculate the final championship standings.
Record the championship winner.

This process is repeated:

10,000 simulations

The frequency with which each driver wins becomes their raw championship probability.

Simulation Structure
Current 2026 Championship State
             │
             ▼
      Remaining Races
             │
             ▼
     Race Performance
       Predictions
             │
             ▼
   Randomized Outcomes
             │
             ▼
      Final Standings
             │
             ▼
    Repeat 10,000 Times
             │
             ▼
 Championship Probabilities
🏁 DNF Modeling

Formula 1 races contain uncertainty from retirements and race incidents.

The simulation therefore incorporates driver DNF rates based on available 2026 race information.

The DNF probability is constrained to a reasonable range within the simulation to prevent extreme values from dominating the championship projections.

🔬 Historical Backtesting

The complete championship simulation was backtested against historical seasons.

The backtesting period was:

2021
2022
2023
2024
2025

For each historical season, the simulation attempted to estimate the championship outcome using information available during the season.

Backtesting Results
Season	Actual Champion	Predicted Champion
2021	Max Verstappen	Lewis Hamilton
2022	Max Verstappen	Max Verstappen
2023	Max Verstappen	Max Verstappen
2024	Max Verstappen	Max Verstappen
2025	Lando Norris	Oscar Piastri
Champion Prediction Accuracy
60%

The historical backtest also provided the data needed to evaluate and calibrate the championship probabilities.

📈 Probability Calibration

Raw Monte Carlo probabilities can sometimes become excessively confident.

For example, a model may assign an extremely high probability to a driver even though historical uncertainty suggests that the probability should be lower.

To address this, the project evaluates temperature scaling for probability calibration.

Leave-One-Season-Out Calibration

Calibration was evaluated using a leave-one-season-out approach.

For each historical season:

One season → Held out
Remaining seasons → Calibration training
Held-out season → Evaluation

This helps avoid evaluating calibration on the same season used to learn the calibration parameter.

Calibration Results
Multiclass Brier Score
Model	Brier Score
Raw	0.6975
Calibrated	0.4554
Log Loss
Model	Log Loss
Raw	1.3210
Calibrated	0.7009

The lower calibrated scores indicate improved probability quality on the leave-one-season-out evaluation.

Final Calibration Parameter

The temperature fitted using the historical seasons was:

Temperature = 5.600

The calibrated probabilities are used for the current 2026 championship estimate.

🏆 Current 2026 Championship Estimate

The current model state is based on the championship standings after Round 15.

The system simulates the remaining 2026 races and applies the calibrated probability transformation.

Current Calibrated Championship Probabilities
Driver	Championship Probability
Andrea Kimi Antonelli	54.72%
George Russell	25.79%
Lewis Hamilton	11.99%
Charles Leclerc	0.39%
Lando Norris	0.39%
Max Verstappen	0.39%
Oscar Piastri	0.39%
Isack Hadjar	0.39%
Other drivers	0.39%

These are model-generated probabilities and are not guarantees of future results.

The probabilities will change as additional 2026 races are completed.

🏎️ Round 16 Race Prediction

The project currently contains a pre-qualifying prediction for the 2026 Round 16 race.

Because this prediction is generated before qualifying, the model does not use the actual current-race grid or qualifying result.

Example Predicted Order
Position	Driver
1	Andrea Kimi Antonelli
2	George Russell
3	Charles Leclerc
4	Lewis Hamilton
5	Oscar Piastri
6	Max Verstappen
7	Isack Hadjar
8	Lando Norris
9	Pierre Gasly
10	Franco Colapinto

The complete prediction is stored in:

data/predicted_race_2026_round_16.csv
📊 Model Feature Importance

The Random Forest model identified constructor and recent performance features as important predictors.

Some of the strongest features included:

constructor_avg_points_last_3
grid
driver_avg_points_last_5
constructor_avg_qualifying_last_3
driver_avg_qualifying_last_5
constructor_championship_points_before

This suggests that both recent constructor competitiveness and driver performance contribute substantially to predicted race finishing position.

🖥️ Streamlit Dashboard

The project includes an interactive Streamlit dashboard.

The dashboard provides:

Current championship probabilities
Top championship contenders
Round 16 race prediction
Raw vs calibrated probabilities
Model performance
Historical backtesting results
Probability calibration results
Project methodology
Dashboard Architecture
CSV Results
     │
     ▼
Streamlit
     │
     ├── Championship Overview
     ├── Probability Chart
     ├── Race Prediction
     ├── Model Evaluation
     ├── Backtesting
     └── Calibration
Run Dashboard Locally
python -m streamlit run app.py
📁 Project Structure
F1-2026-Championship-Prediction/
│
├── README.md
├── app.py
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── calibrated_historical_probabilities.csv
│   ├── constructor_standings_2018_2025.csv
│   ├── constructor_standings_2026.csv
│   ├── driver_standings_2018_2025.csv
│   ├── driver_standings_2026.csv
│   ├── loo_calibration_results.csv
│   ├── loo_calibration_summary.csv
│   ├── monte_carlo_2026_calibrated_results.csv
│   ├── monte_carlo_2026_results.csv
│   ├── monte_carlo_backtest_driver_results.csv
│   ├── monte_carlo_backtest_summary.csv
│   ├── predicted_race_2026_round_16.csv
│   ├── probability_calibration_summary.csv
│   ├── qualifying_results_2018_2025.csv
│   ├── qualifying_results_2026.csv
│   ├── race_results_2018_2025.csv
│   ├── race_results_2026.csv
│   ├── sprint_results_2018_2025.csv
│   └── sprint_results_2026.csv
│
└── src/
    ├── add_championship_features.py
    ├── analyze_feature_importance.py
    ├── backtest_monte_carlo.py
    ├── build_2026_features.py
    ├── calibrate_championship_probabilities.py
    ├── calibrate_championship_probabilities_loo.py
    ├── collect_2026_data.py
    ├── constructor_standings_collection.py
    ├── create_master_dataset.py
    ├── data_collection.py
    ├── driver_standings_collection.py
    ├── feature_engineering.py
    ├── feature_engineering_v2.py
    ├── monte_carlo_simulation.py
    ├── predict_2026_race.py
    ├── prepare_ml_dataset.py
    ├── qualifying_collection.py
    ├── rebuild_championship_features.py
    ├── sprint_collection.py
    ├── train_baseline_model.py
    ├── train_gradient_boosting_model.py
    ├── train_prequalifying_model.py
    └── tune_random_forest.py
🛠️ Technologies Used
Programming
Python
Data Processing
Pandas
NumPy
Machine Learning
Scikit-learn
Random Forest Regression
Visualization
Plotly
Streamlit
Data & Modeling Techniques
Feature Engineering
Time-Based Validation
Monte Carlo Simulation
Historical Backtesting
Probability Calibration
Temperature Scaling
Development
Git
GitHub
REST API
⚙️ Installation
1. Clone the Repository
git clone https://github.com/Jo-Avi/F1-2026-Championship-Prediction.git

Move into the project directory:

cd F1-2026-Championship-Prediction
2. Create a Virtual Environment
python -m venv .venv
Windows
.venv\Scripts\Activate.ps1
3. Install Dependencies
pip install -r requirements.txt
▶️ Running the Project
Run the Streamlit Dashboard
python -m streamlit run app.py
Run the Data Collection Pipeline

Historical data collection scripts can be executed from the src/ directory.

Example:

python src/data_collection.py

Other collection scripts include:

qualifying_collection.py
sprint_collection.py
driver_standings_collection.py
constructor_standings_collection.py
Run Feature Engineering
python src/feature_engineering.py

Additional feature engineering stages are available through:

feature_engineering_v2.py
add_championship_features.py
rebuild_championship_features.py
Train the Model

The pre-qualifying Random Forest model can be trained with:

python src/train_prequalifying_model.py
Generate Race Prediction
python src/predict_2026_race.py
Run Monte Carlo Simulation
python src/monte_carlo_simulation.py
Run Historical Backtesting
python src/backtest_monte_carlo.py
Run Probability Calibration
python src/calibrate_championship_probabilities_loo.py
🔄 Updating the Model During the 2026 Season

The project is designed to become a rolling championship prediction system.

After each completed race:

Completed Race
      ↓
Collect Latest Results
      ↓
Update Driver & Constructor Standings
      ↓
Rebuild Features
      ↓
Generate Next Race Prediction
      ↓
Run Monte Carlo Simulation
      ↓
Calibrate Probabilities
      ↓
Update Dashboard

For Sprint weekends, Sprint results can also be incorporated into the championship simulation.

🔮 Future Improvements

Potential improvements include:

Race Prediction
Circuit-specific performance
Practice session performance
Weather conditions
Track characteristics
Tire strategy
Pit-stop performance
Qualifying/grid information when available
Machine Learning
Gradient boosting models
XGBoost / LightGBM comparison
Learning-to-rank approaches
Ensemble models
Probabilistic finishing-position models
Simulation
More sophisticated DNF modeling
Driver-specific uncertainty
Constructor-specific uncertainty
Strategy uncertainty
Correlated driver outcomes
Dashboard
Championship progression over time
Driver head-to-head comparisons
Constructor championship probabilities
Race-by-race probability changes
Interactive feature importance
Simulation distribution charts
Automation
Automatic F1 data updates
Automatic model retraining
Automatic post-race predictions
Automated dashboard updates
⚠️ Limitations

The model has several limitations.

Limited Historical Sample

The model is based on F1 seasons from 2018 onward, which limits the number of historical championship outcomes available for calibration.

Race Outcome Uncertainty

F1 races contain unpredictable events such as:

Mechanical failures
Collisions
Safety cars
Weather changes
Strategy decisions
Penalties

Not all of these factors are represented in the current model.

Pre-Qualifying Prediction

The current Round 16 prediction is generated before the actual qualifying result.

Therefore, it does not benefit from the actual starting grid for that race.

Probability Interpretation

The championship probabilities represent simulated model outcomes rather than guaranteed forecasts.

📜 Disclaimer

This project is an educational machine learning and data science project.

The championship probabilities are generated from historical data, engineered features, machine learning predictions, uncertainty assumptions, and Monte Carlo simulations.

They should not be interpreted as guaranteed outcomes or used as a substitute for official Formula 1 results.

👨‍💻 Author
Aviral Yadav

B.Tech in Computer Science and Engineering
Specialization: Cyber Security and Digital Forensics
VIT Bhopal

Links
GitHub: https://github.com/Jo-Avi
Project Repository: https://github.com/Jo-Avi/F1-2026-Championship-Prediction
⭐ Project Highlights
✓ Historical F1 data from 2018–2025
✓ Driver & constructor feature engineering
✓ Leakage-aware time-based validation
✓ Random Forest race-position prediction
✓ Pre-qualifying race prediction
✓ 10,000-run Monte Carlo simulation
✓ Historical backtesting from 2021–2025
✓ Leave-one-season-out calibration
✓ Temperature-scaled championship probabilities
✓ Interactive Streamlit dashboard
✓ Designed for rolling 2026 season updates
🏁 Final Note

The goal of this project is not simply to predict a single F1 champion.

It is to build a complete race-level probabilistic forecasting pipeline that can evolve throughout a Formula 1 season as new race results become available.


### After you save it

Run:

```powershell
git add README.md
git commit -m "Add comprehensive project README"
git push