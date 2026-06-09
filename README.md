# 🏏 IPL Win Probability Predictor

## Overview

Cricket matches can change dramatically within a few overs. This project aims to estimate the winning probability of a team during an IPL run chase using Machine Learning.

The application takes the current match situation as input and predicts the chances of the batting team winning the match in real time. The prediction is powered by an XGBoost model trained on historical IPL match data.

The final application is deployed as an interactive Streamlit web app where users can select teams, venue, target score, current score, overs completed, and wickets lost to generate live winning probabilities.

---

## Problem Statement

During a run chase, factors such as the target score, wickets remaining, run rate, and venue conditions influence the outcome of the match.

The objective of this project is to build a machine learning model capable of estimating the probability of victory for the chasing team based on the current state of the game.

---

## Dataset

The model was trained using historical IPL match and ball-by-ball datasets.

The data includes:

* Match information
* Batting and bowling teams
* Venue details
* Runs scored
* Wickets lost
* Ball-by-ball events

After data cleaning and preprocessing, relevant features were extracted for model training.

---

## Feature Engineering

The following features were used to train the model:

* Batting Team
* Bowling Team
* Venue
* Target Score
* Runs Left
* Balls Left
* Wickets Left
* Current Run Rate (CRR)
* Required Run Rate (RRR)

These features help capture the current match situation and provide meaningful information for prediction.

---

## Models Evaluated

Three different machine learning models were tested and compared:

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 80.68%   |
| Random Forest       | 94.39%   |
| XGBoost             | 96.28%   |

### Final Model

XGBoost was selected as the final model because it achieved the highest accuracy and produced more reliable probability estimates compared to the other approaches.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Streamlit
* Plotly

---

## Application Features

* Interactive web interface built with Streamlit
* Real-time IPL win probability prediction
* Match statistics display
* Probability visualization
* User-friendly dashboard
* XGBoost-powered prediction engine

---

## Project Structure

```text
IPL_Win_Probability_Predictor/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
├── models/
│   ├── xgb_pipe.pkl
│   ├── teams.pkl
│   └── venues.pkl
│
└── notebooks/
    └── data_understanding.ipynb
```

---

## How to Run

1. Clone the repository

```bash
git clone <repository-url>
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Launch the application

```bash
streamlit run app.py
```

---

## Author

**Sarthak Jain**

B.Tech (Artificial Intelligence)
Teerthanker Mahaveer University

Passionate about Machine Learning, Data Science, and AI-driven applications.
