import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

st.markdown("""
<style>

.main {
    background: linear-gradient(
        135deg,
        #0f172a,
        #111827
    );
}

.stButton>button {
    background-color:#00D4FF;
    color:black;
    font-weight:bold;
    border-radius:10px;
    height:55px;
}

.stMetric {
    background-color:#1e293b;
    padding:10px;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="IPL Win Probability Predictor",
    page_icon="🏏",
    layout="wide"
)

st.markdown("""
<div style='text-align:center;padding:20px'>
    <h1 style='color:#00D4FF;'>🏏 IPL Win Probability Predictor</h1>
    <h4>Live Match Winning Probability using XGBoost Machine Learning Model</h4>
</div>
""", unsafe_allow_html=True)

model = pickle.load(open('models/xgb_pipe.pkl','rb'))
teams = pickle.load(open('models/teams.pkl','rb'))
venues = pickle.load(open('models/venues.pkl','rb'))

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Batting Team", teams)

    venue = st.selectbox("Venue", venues)

    target = st.number_input(
        "Target",
        min_value=1,
        step=1
    )

    overs_completed = st.number_input(
        "Overs Completed",
        min_value=0.0,
        max_value=20.0,
        step=0.1
    )

with col2:
    bowling_team = st.selectbox("Bowling Team", teams)

    current_score = st.number_input(
        "Current Score",
        min_value=0,
        step=1
    )

    wickets_out = st.number_input(
        "Wickets Lost",
        min_value=0,
        max_value=10,
        step=1
    )

st.markdown("---")

predict = st.button(
    "🚀 Predict Winning Probability",
    use_container_width=True
)

if predict:
    runs_left = target - current_score
    balls_left = 120 - (overs_completed * 6)
    wickets_left = 10 - wickets_out

    crr = current_score / max(overs_completed, 0.1)
    rrr = (runs_left * 6) / balls_left

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'venue': [venue],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets_left': [wickets_left],
        'target': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Runs Left", runs_left)
    col2.metric("Balls Left", balls_left)
    col3.metric("CRR", round(crr,2))
    col4.metric("RRR", round(rrr,2))

    result = model.predict_proba(input_df)

    loss = result[0][0]
    win = result[0][1]

    st.subheader("📊 Win Probability")

    st.metric(
        label=f"{batting_team}",
        value=f"{round(win*100)}%"
    )

    st.progress(float(win))

    st.metric(
        label=f"{bowling_team}",
        value=f"{round(loss*100)}%"
    )

    st.progress(float(loss))

    chart_df = pd.DataFrame({
    "Team":[batting_team,bowling_team],
    "Probability":[win*100,loss*100]
    })

    fig = px.pie(
        chart_df,
        values="Probability",
        names="Team",
        hole=0.5,
        title="Winning Probability Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)