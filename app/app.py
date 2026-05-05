import sys
import os
import streamlit as st
import pandas as pd
import joblib
import altair as alt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#Streamlit Page Configuration
st.set_page_config(
    page_title="Los Angeles Lakers (NBA) Season Wins Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏀 Los Angeles Lakers (NBA) Season Wins Predictor")
st.markdown(
    """
Predict the number of wins for the Los Angeles Lakers based on season statistics.
Select the year to predict future wins and see historical trends.
"""
)

#Load Model and Data
model_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "xgb_future_wins_model.pkl")
)
model = joblib.load(model_path)

parquet_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "processed", "season_features.parquet")
)
df = pd.read_parquet(parquet_path)

#Lagged features
lag_features = ['avg_pts','avg_fg_pct','avg_3p_pct','avg_ft_pct',
                'avg_efg_pct','avg_ast','avg_reb','avg_stl',
                'avg_blk','avg_tov']

for col in lag_features:
    if f'prev_{col}' not in df.columns:
        df[f'prev_{col}'] = df[col].shift(1)

df = df.dropna(subset=[f'prev_{col}' for col in lag_features]).reset_index(drop=True)

#Sidebar: Year Selection
st.sidebar.header("Select Season")
year = st.sidebar.slider(
    "Year",
    min_value=int(df['year'].min()),
    max_value=int(df['year'].max()) + 1,
    value=int(df['year'].max())
)

feature_cols = [f'prev_{col}' for col in lag_features]

#Display Input Features
st.subheader(f"Lag Features for Year {int(year)}")

if year in df['year'].values:
    features_row = df[df['year'] == year][feature_cols]
else:
    features_row = df.iloc[[-1]][feature_cols]
    st.info("Using most recent season's lag features for future year prediction.")

st.dataframe(features_row.T.rename(columns={features_row.index[0]: "Value"}))

#Prediction Button
if st.button("Predict Wins"):
    pred = model.predict(features_row)
    st.success(f"Predicted Wins for {int(year)}: {pred[0]:.1f}")

#Historical Wins Chart
st.subheader("Historical Wins")

history_df = pd.DataFrame({
    "Year": df['year'],
    "Wins": df['wins']
})

chart = alt.Chart(history_df).mark_line(point=True).encode(
    x='Year:O',
    y='Wins:Q',
    tooltip=['Year', 'Wins']
).properties(
    width=800,
    height=400
).interactive()

st.altair_chart(chart, use_container_width=True)
