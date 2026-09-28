🏎️ F1 2026 Championship Prediction

An end-to-end machine learning project that estimates Formula 1 2026
World Drivers' Championship probabilities using historical race data,
driver and constructor performance features, Random Forest regression,
Monte Carlo simulation, historical backtesting, and probability
calibration.

Current model state: After Round 15 of the 2026 season
Simulation: 10,000 championship simulations
Primary model: Random Forest Regressor

📌 Overview

Instead of directly predicting the championship winner, the project
breaks the problem into race-level predictions:

Historical F1 Data
        ↓
Data Cleaning & Feature Engineering
        ↓
Random Forest Race-Position Model
        ↓
Upcoming Race Prediction
        ↓
Monte Carlo Simulation
        ↓
Historical Backtesting
        ↓
Probability Calibration
        ↓
2026 Championship Probabilities
        ↓
Streamlit Dashboard

The system is designed as a rolling prediction pipeline that can be
updated after each completed race.

🎯 Objectives

Predict driver finishing positions for upcoming races.

Capture recent driver and constructor performance.

Incorporate championship standings before each race.

Reduce future-data leakage through chronological feature
engineering.

Evaluate the model using time-based validation.

Simulate thousands of possible championship outcomes.

Estimate championship-winning probabilities.

Backtest the simulation against historical F1 seasons.

Calibrate championship probabilities using out-of-sample historical
results.

Present results through an interactive Streamlit dashboard.

📊 Data

Historical Formula 1 data from 2018--2025 is used for model
development and evaluation.

The project works with:

Race results

Qualifying results

Sprint results

Driver standings

Constructor standings

The 2026 dataset is updated as the season progresses.

Data Source

Historical and 2026 race data is collected through the Jolpica-F1
API, an Ergast-compatible Formula 1 data API.

🧠 Feature Engineering

The model combines recent driver performance, constructor performance,
and championship-state information.

Driver Features

Average finishing position over the last 3 races

Average finishing position over the last 5 races

Average points over the last 3 races

Average points over the last 5 races

Average qualifying position over the last 3 races

Average qualifying position over the last 5 races

Historical DNF rate

Constructor Features

Average finishing position over the last 3 races

Average points over the last 3 races

Average qualifying position over the last 3 races

Championship Features

Driver championship position before the race

Driver championship points before the race

Driver wins before the race

Constructor championship position before the race

Constructor championship points before the race

Constructor wins before the race

Rolling features are shifted so that information from the target race is
not used to construct its own prediction.

🤖 Machine Learning

Random Forest Regression

The primary model predicts a driver's finishing position rather than
directly predicting the championship winner.

This provides many driver-race observations for model training instead
of only one champion per season.

Model Configuration

Parameter                                      Value

Model                        Random Forest Regressor
Trees                                            300
Maximum Depth                                     12
Minimum Samples per Leaf                           3
Random State                                      42

🧪 Time-Based Validation

Because Formula 1 data is chronological, a random train/test split was
avoided.

Dataset      Seasons

Training     2018--2023
Validation   2024
Test         2025

Pre-Qualifying Model Performance

Dataset                       MAE        RMSE

Training                    2.374       3.040
Validation --- 2024         3.485       4.364
Test --- 2025           3.853   4.758

🎲 Monte Carlo Championship Simulation

A single race prediction cannot represent the uncertainty of an entire
F1 championship. The project therefore uses 10,000 Monte Carlo
simulations.

For each simulation:

Start from the current championship standings.

Predict the remaining race outcomes.

Introduce uncertainty around predicted finishing positions.

Account for DNF probability.

Convert finishing positions into championship points.

Simulate the remaining races.

Calculate the final championship standings.

Record the championship winner.

Current Championship State
          ↓
Remaining Races
          ↓
Predicted Race Performance
          ↓
Uncertainty + DNF Modeling
          ↓
Final Championship Standings
          ↓
Repeat 10,000 Times
          ↓
Championship Probabilities

🔬 Historical Backtesting

The complete championship simulation was backtested on the
2021--2025 seasons.

Season   Actual Champion   Predicted Champion

2021     Max Verstappen    Lewis Hamilton
2022     Max Verstappen    Max Verstappen
2023     Max Verstappen    Max Verstappen
2024     Max Verstappen    Max Verstappen
2025     Lando Norris      Oscar Piastri

Champion Prediction Accuracy

60%

Backtesting was also used to evaluate the quality of the raw
championship probabilities.

📈 Probability Calibration

Raw Monte Carlo probabilities can become overly confident. The project
therefore evaluates temperature scaling using a leave-one-season-out
approach.

Out-of-Sample Calibration Results

Multiclass Brier Score

Model          Brier Score

Raw                 0.6975
Calibrated      0.4554

Log Loss

Model            Log Loss

Raw                1.3210
Calibrated     0.7009

Final Calibration Parameter

Temperature = 5.600

🏆 Current 2026 Championship Estimate

The current model state is based on the championship standings after
Round 15.

Driver                    Championship Probability

Andrea Kimi Antonelli                   54.72%
George Russell                          25.79%
Lewis Hamilton                          11.99%
Charles Leclerc                              0.39%
Lando Norris                                 0.39%
Max Verstappen                               0.39%
Oscar Piastri                                0.39%

The remaining drivers each receive approximately 0.39% in the
current calibrated output.

These are model-generated probabilities, not guaranteed outcomes. They
will change as the 2026 season progresses.

🏁 Round 16 Prediction

The project includes a pre-qualifying prediction for 2026 Round 16.

Because the prediction is generated before the target race's qualifying
session, the current-race grid and qualifying result are not used.

Position Driver

       1 Andrea Kimi Antonelli
       2 George Russell
       3 Charles Leclerc
       4 Lewis Hamilton
       5 Oscar Piastri
       6 Max Verstappen
       7 Isack Hadjar
       8 Lando Norris
       9 Pierre Gasly
      10 Franco Colapinto

Full prediction:

data/predicted_race_2026_round_16.csv

📊 Model Insights

The Random Forest feature analysis identified several strong predictors,
including:

constructor_avg_points_last_3

grid

driver_avg_points_last_5

constructor_avg_qualifying_last_3

driver_avg_qualifying_last_5

constructor_championship_points_before

These features capture recent constructor competitiveness, starting
position, recent driver performance, and championship state.

🖥️ Streamlit Dashboard

The project includes a Streamlit dashboard for exploring:

Championship probabilities

Top contenders

Round 16 race prediction

Raw vs calibrated probabilities

Model performance

Historical backtesting

Probability calibration

Project methodology

Run the Dashboard

python -m streamlit run app.py

📁 Project Structure

F1-2026-Championship-Prediction/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── Historical race data
│   ├── 2026 race data
│   ├── Monte Carlo results
│   ├── Backtesting results
│   └── Calibration results
│
└── src/
    ├── Data collection
    ├── Feature engineering
    ├── Model training
    ├── Race prediction
    ├── Monte Carlo simulation
    ├── Historical backtesting
    └── Probability calibration

🛠️ Tech Stack

Category           Technologies

Language           Python
Data Processing    Pandas, NumPy
Machine Learning   Scikit-learn
Model              Random Forest Regressor
Simulation         Monte Carlo
Visualization      Plotly
Dashboard          Streamlit
Data Source        Jolpica-F1 API
Version Control    Git, GitHub

⚙️ Installation

1. Clone the Repository

git clone https://github.com/Jo-Avi/F1-2026-Championship-Prediction.git
cd F1-2026-Championship-Prediction

2. Create a Virtual Environment

python -m venv .venv

3. Activate the Environment

Windows

.venv\Scripts\Activate.ps1

macOS / Linux

source .venv/bin/activate

4. Install Dependencies

pip install -r requirements.txt

▶️ Running the Project

Dashboard

python -m streamlit run app.py

Train the Pre-Qualifying Model

python src/train_prequalifying_model.py

Generate a Race Prediction

python src/predict_2026_race.py

Run the Monte Carlo Simulation

python src/monte_carlo_simulation.py

Run Historical Backtesting

python src/backtest_monte_carlo.py

Run Leave-One-Season-Out Calibration

python src/calibrate_championship_probabilities_loo.py

🔄 Updating the Model During the 2026 Season

The project is designed to support rolling updates:

Completed Race
      ↓
Collect Latest Results
      ↓
Update Standings
      ↓
Rebuild Features
      ↓
Predict Next Race
      ↓
Run Monte Carlo Simulation
      ↓
Calibrate Probabilities
      ↓
Update Dashboard

For Sprint weekends, Sprint results can also be incorporated into the
championship simulation.

🔮 Future Improvements

Circuit-specific performance

Practice-session performance

Weather conditions

Track characteristics

Tire strategy

Pit-stop performance

Qualifying/grid information when available

Gradient boosting and ensemble models

More sophisticated DNF modeling

Driver- and constructor-specific uncertainty

Championship progression visualizations

Automated F1 data updates

Automated post-race predictions

⚠️ Limitations

The historical calibration sample contains a limited number of
completed championship seasons.

F1 outcomes contain unpredictable events such as incidents,
mechanical failures, safety cars, penalties, weather, and strategy
decisions.

The current Round 16 prediction is generated before qualifying and
therefore does not use the actual starting grid.

Detailed telemetry, tire strategy, weather, and practice-session
performance are not yet included.

Championship probabilities are model estimates and are not
guarantees.

📜 Disclaimer

This project is an educational machine learning and data science
project.

The championship probabilities are generated from historical data,
engineered features, machine learning predictions, uncertainty
assumptions, and Monte Carlo simulations. They are not official Formula
1 predictions and do not guarantee future race or championship outcomes.

👨‍💻 Author

Aviral Yadav

B.Tech in Computer Science and Engineering
Specialization: Cyber Security and Digital Forensics
VIT Bhopal

GitHub: Jo-Avi

Repository: F1 2026 Championship
Prediction

🏁 Built to turn race-level predictions into a probabilistic championship forecast.