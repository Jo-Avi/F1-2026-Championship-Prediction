# F1 2026 Championship Prediction

A machine learning project that predicts Formula 1 race finishing positions and estimates 2026 World Drivers' Championship probabilities using historical F1 data, Random Forest regression, Monte Carlo simulation, historical backtesting, and probability calibration.

---

## 🚀 Live Dashboard

**Coming Soon**

---

## 📊 Project Overview

The project uses a race-by-race prediction approach instead of directly predicting the championship winner.

The overall workflow is:

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
```

The system is designed to be updated after each completed race during the 2026 season.

---

## 🎯 Features

### 🏎️ Race Prediction

* Predicts driver finishing positions for upcoming races
* Uses recent driver performance
* Uses recent constructor performance
* Incorporates championship standings
* Supports pre-qualifying race predictions

### 📊 Feature Engineering

* Driver rolling performance features
* Constructor rolling performance features
* Qualifying performance
* Driver DNF rate
* Driver championship position and points
* Driver wins before the race
* Constructor championship position and points
* Constructor wins before the race

### 🎲 Monte Carlo Simulation

* Simulates the remaining championship 10,000 times
* Uses predicted race finishing positions
* Introduces prediction uncertainty
* Incorporates DNF probability
* Converts race results into championship points
* Estimates championship-winning probabilities

### 🔬 Historical Backtesting

The championship simulation is evaluated against previous F1 seasons:

* 2021
* 2022
* 2023
* 2024
* 2025

### 📈 Probability Calibration

The project uses leave-one-season-out calibration and temperature scaling to evaluate and improve the reliability of the raw Monte Carlo championship probabilities.

---

## 📚 Dataset

Historical Formula 1 data from **2018–2025** is used for model development and evaluation.

The project includes:

* Race results
* Qualifying results
* Sprint results
* Driver standings
* Constructor standings

2026 data is maintained separately and updated as the season progresses.

### Data Source

F1 data is collected using the **Jolpica-F1 API**.

---

## 🤖 Machine Learning Model

The primary race prediction model is a **Random Forest Regressor**.

Instead of directly predicting the championship winner, the model predicts the expected finishing position of each driver in an upcoming race.

### Model Configuration

| Parameter | Value |
| --------- | ----- |
| Model | Random Forest Regressor |
| Trees | 300 |
| Maximum Depth | 12 |
| Minimum Samples per Leaf | 3 |
| Random State | 42 |

---

## 🧠 Feature Engineering

The model combines driver, constructor, qualifying, and championship-state information.

### Driver Features

* Average finishing position over the last 3 races
* Average finishing position over the last 5 races
* Average points over the last 3 races
* Average points over the last 5 races
* Average qualifying position over the last 3 races
* Average qualifying position over the last 5 races
* Historical DNF rate

### Constructor Features

* Average finishing position over the last 3 races
* Average points over the last 3 races
* Average qualifying position over the last 3 races

### Championship Features

* Driver championship position before the race
* Driver championship points before the race
* Driver wins before the race
* Constructor championship position before the race
* Constructor championship points before the race
* Constructor wins before the race

Rolling features are calculated using information available before the target race to reduce future-data leakage.

---

## 🧪 Model Evaluation

The project uses chronological validation instead of a random train/test split.

```text
Training  → 2018–2023
Validation → 2024
Testing   → 2025
```

### Pre-Qualifying Model Performance

| Dataset | MAE | RMSE |
| ------- | ---: | ----: |
| Training | 2.374 | 3.040 |
| Validation — 2024 | 3.485 | 4.364 |
| Test — 2025 | **3.853** | **4.758** |

---

## 🎲 Monte Carlo Simulation

A single race prediction cannot represent the uncertainty of an entire championship.

The project therefore runs **10,000 Monte Carlo simulations**.

For each simulation:

1. Start from the current championship standings
2. Predict the remaining race outcomes
3. Add prediction uncertainty
4. Account for possible DNFs
5. Convert finishing positions into championship points
6. Calculate the final championship standings
7. Record the championship winner

```text
Current Championship State
          ↓
Remaining Races
          ↓
Race Predictions
          ↓
Uncertainty + DNF Modeling
          ↓
Final Championship Standings
          ↓
Repeat 10,000 Times
          ↓
Championship Probabilities
```

---

## 🔬 Historical Backtesting

The championship simulation was backtested against the 2021–2025 seasons.

| Season | Actual Champion | Predicted Champion |
| ------ | --------------- | ------------------ |
| 2021 | Max Verstappen | Lewis Hamilton |
| 2022 | Max Verstappen | Max Verstappen |
| 2023 | Max Verstappen | Max Verstappen |
| 2024 | Max Verstappen | Max Verstappen |
| 2025 | Lando Norris | Oscar Piastri |

### Champion Prediction Accuracy

**60%**

The backtest was also used to evaluate the reliability of the championship probabilities.

---

## 📈 Probability Calibration

Raw Monte Carlo probabilities can become overly confident.

The project evaluates **temperature scaling** using a leave-one-season-out approach.

### Multiclass Brier Score

| Model | Brier Score |
| ----- | -----------: |
| Raw | 0.6975 |
| Calibrated | **0.4554** |

### Log Loss

| Model | Log Loss |
| ----- | --------: |
| Raw | 1.3210 |
| Calibrated | **0.7009** |

### Final Calibration Temperature

```text
5.600
```

---

## 🏆 Current 2026 Championship Estimate

The current model state is based on the championship standings after **Round 15**.

| Driver | Championship Probability |
| ------ | ------------------------: |
| Andrea Kimi Antonelli | **54.72%** |
| George Russell | **25.79%** |
| Lewis Hamilton | **11.99%** |
| Charles Leclerc | 0.39% |
| Lando Norris | 0.39% |
| Max Verstappen | 0.39% |
| Oscar Piastri | 0.39% |

The remaining drivers receive approximately 0.39% each in the current calibrated output.

> These are model-generated probabilities and will change as additional 2026 race results become available.

---

## 🏁 Round 16 Prediction

The project currently includes a **pre-qualifying prediction** for Round 16.

The current-race grid and qualifying result are not used for this prediction.

### Predicted Top 10

| Position | Driver |
| -------- | ------ |
| 1 | Andrea Kimi Antonelli |
| 2 | George Russell |
| 3 | Charles Leclerc |
| 4 | Lewis Hamilton |
| 5 | Oscar Piastri |
| 6 | Max Verstappen |
| 7 | Isack Hadjar |
| 8 | Lando Norris |
| 9 | Pierre Gasly |
| 10 | Franco Colapinto |

The complete prediction is available in:

```text
data/predicted_race_2026_round_16.csv
```

---

## 📊 Model Insights

Some of the strongest Random Forest features include:

* `constructor_avg_points_last_3`
* `grid`
* `driver_avg_points_last_5`
* `constructor_avg_qualifying_last_3`
* `driver_avg_qualifying_last_5`
* `constructor_championship_points_before`

These features capture recent constructor competitiveness, starting position, recent driver performance, and championship state.

---

## 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit dashboard for displaying:

* Championship probabilities
* Top championship contenders
* Round 16 prediction
* Raw vs calibrated probabilities
* Model performance
* Historical backtesting
* Probability calibration
* Project methodology

### Run the Dashboard

```bash
python -m streamlit run app.py
```

---

## 🛠️ Tech Stack

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Random Forest Regressor

### Simulation & Statistics

* Monte Carlo Simulation
* Temperature Scaling
* Leave-One-Season-Out Calibration

### Visualization & Dashboard

* Plotly
* Streamlit

### Data Source

* Jolpica-F1 API

### Development

* Git
* GitHub

---

## 📂 Project Structure

```text
F1-2026-Championship-Prediction/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── Historical F1 datasets
│   ├── 2026 datasets
│   ├── Monte Carlo results
│   ├── Backtesting results
│   └── Calibration results
│
└── src/
    ├── Data collection scripts
    ├── Feature engineering scripts
    ├── Model training scripts
    ├── Race prediction
    ├── Monte Carlo simulation
    ├── Historical backtesting
    └── Probability calibration
```

---

## ⚙️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Jo-Avi/F1-2026-Championship-Prediction.git
cd F1-2026-Championship-Prediction
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Environment

#### Windows

```powershell
.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Run the Dashboard

```bash
python -m streamlit run app.py
```

### Train the Pre-Qualifying Model

```bash
python src/train_prequalifying_model.py
```

### Generate a Race Prediction

```bash
python src/predict_2026_race.py
```

### Run Monte Carlo Simulation

```bash
python src/monte_carlo_simulation.py
```

### Run Historical Backtesting

```bash
python src/backtest_monte_carlo.py
```

### Run Probability Calibration

```bash
python src/calibrate_championship_probabilities_loo.py
```

---

## 🔄 Updating the Project During the 2026 Season

The project is designed to be updated after each completed race.

```text
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
```

For Sprint weekends, Sprint results can also be incorporated into the championship simulation.

---

## 🔮 Future Improvements

* Circuit-specific performance
* Practice-session performance
* Weather conditions
* Track characteristics
* Tire strategy
* Pit-stop performance
* Qualifying/grid information
* More advanced DNF modeling
* Driver-specific uncertainty
* Constructor-specific uncertainty
* Championship progression charts
* Automated F1 data updates
* Automated post-race predictions
* Automated dashboard updates

---

## ⚠️ Limitations

* The historical calibration sample contains a limited number of championship seasons.
* F1 races contain unpredictable events such as incidents, mechanical failures, safety cars, penalties, weather, and strategy decisions.
* The current Round 16 prediction is generated before qualifying and does not use the actual starting grid.
* Detailed telemetry, weather, practice-session performance, and race strategy are not yet included.
* Championship probabilities are model estimates and are not guaranteed outcomes.

---

## 📜 Disclaimer

This project is intended for educational and portfolio purposes.

The championship probabilities are generated using historical data, machine learning predictions, uncertainty assumptions, and Monte Carlo simulations. They are not official Formula 1 predictions and should not be interpreted as guaranteed future results.

---

## 👨‍💻 Author

**Aviral Yadav**

B.Tech in Computer Science and Engineering  
Specialization: Cyber Security and Digital Forensics  
VIT Bhopal

* GitHub: [Jo-Avi](https://github.com/Jo-Avi)
* Project: [F1 2026 Championship Prediction](https://github.com/Jo-Avi/F1-2026-Championship-Prediction)

---

## ⭐ Project Highlights

* Historical F1 data from **2018–2025**
* Driver and constructor feature engineering
* Chronological model validation
* Random Forest race-position prediction
* Pre-qualifying race prediction
* **10,000-run Monte Carlo simulation**
* Historical backtesting from **2021–2025**
* Leave-one-season-out probability calibration
* Temperature-scaled championship probabilities
* Interactive Streamlit dashboard
* Designed for rolling updates throughout the 2026 season