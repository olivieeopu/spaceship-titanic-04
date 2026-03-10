"""
Session 04 – Streamlit App
Serve the trained Spaceship Titanic Logistic Regression model
Run with: streamlit run app_streamlit.py
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

from preprocessing import feature_engineering, preprocess_data

# load model
model = joblib.load(Path(__file__).parent/"model/logistic_model.pkl")


def main():

    st.title("ASG 04 MD - Fransiska - Spaceship Titanic Model Deployment")

    st.write("Enter passenger information")

    # 13 INPUT FEATURES

    HomePlanet = st.selectbox("HomePlanet", ["Earth","Europa","Mars"])
    CryoSleep = st.selectbox("CryoSleep", [False, True])
    Destination = st.selectbox("Destination", ["TRAPPIST-1e","PSO J318.5-22","55 Cancri e"])
    VIP = st.selectbox("VIP", [False, True])

    Age = st.slider("Age", 0, 100, 30)

    RoomService = st.number_input("RoomService", value=0)
    FoodCourt = st.number_input("FoodCourt", value=0)
    ShoppingMall = st.number_input("ShoppingMall", value=0)
    Spa = st.number_input("Spa", value=0)
    VRDeck = st.number_input("VRDeck", value=0)

    Deck = st.selectbox("Deck", ["A","B","C","D","E","F","G","Unknown"])
    Side = st.selectbox("Side", ["P","S","Unknown"])
    Age_group = st.selectbox("Age Group", ["Child","Teen","Young_Adult","Adult","Senior"])

    if st.button("Make Prediction"):

        features = pd.DataFrame({
            "HomePlanet":[HomePlanet],
            "CryoSleep":[CryoSleep],
            "Destination":[Destination],
            "VIP":[VIP],
            "Age":[Age],
            "RoomService":[RoomService],
            "FoodCourt":[FoodCourt],
            "ShoppingMall":[ShoppingMall],
            "Spa":[Spa],
            "VRDeck":[VRDeck],
            "Deck":[Deck],
            "Side":[Side],
            "Age_group":[Age_group]
        })

        result = make_prediction(features)

        if result == 1:
            st.success("Passenger was Transported")
        else:
            st.error("Passenger was NOT Transported")


def make_prediction(features):

    # dummy columns needed for feature engineering
    features["PassengerId"] = "0001_01"
    features["Cabin"] = "F/123/P"
    features["Name"] = "John Doe"

    # apply same pipeline as training
    df = feature_engineering(features)

    X, _ = preprocess_data(df, is_train=False)

    prediction = model.predict(X)

    return prediction[0]


if __name__ == "__main__":
    main()
