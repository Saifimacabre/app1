
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load datasets
@st.cache_data
def load_data():
    fert_df = pd.read_csv("F:\\ML pros\\Fertilizer Prediction.csv")
    crop_df = pd.read_csv("F:\\ML pros\\Crop_recommendation.csv")
    return fert_df, crop_df

fertilizer_df, crop_recommendation_df = load_data()

# Encode non-numeric columns
fert_encoder = LabelEncoder()
for col in fertilizer_df.select_dtypes(include=["object"]).columns:
    fertilizer_df[col] = fert_encoder.fit_transform(fertilizer_df[col])

crop_encoder = LabelEncoder()
for col in crop_recommendation_df.select_dtypes(include=["object"]).columns:
    crop_recommendation_df[col] = crop_encoder.fit_transform(crop_recommendation_df[col])

# UI Title
st.title("🌾 Crop and Fertilizer Prediction App")

# User Choice
choice = st.selectbox("What do you want to predict?", ["Crop", "Fertilizer"])

# Input Fields
st.subheader("Enter the following parameters:")

temp = st.number_input("🌡️ Temperature")
humidity = st.number_input("💧 Humidity")
moisture = st.number_input("🧪 Moisture")
soil_type = st.text_input("🌱 Soil Type (label-encoded integer)")
crop_type = st.text_input("🌾 Crop Type (label-encoded integer)")
nitrogen = st.number_input("🧬 Nitrogen")
potassium = st.number_input("🧪 Potassium")
phosphorous = st.number_input("🧪 Phosphorous")
ph = st.number_input("pH")
rainfall = st.number_input("🌧️ Rainfall")

# Prediction logic
if st.button("🔍 Predict"):
    if choice == "Crop":
        X_crop = crop_recommendation_df.drop('label', axis=1)
        y_crop = crop_recommendation_df['label']
        crop_model = RandomForestClassifier()
        crop_model.fit(X_crop, y_crop)

        input_df = pd.DataFrame([[nitrogen, phosphorous, potassium, temp, humidity, ph, rainfall]],
                                columns=X_crop.columns)
        prediction = crop_model.predict(input_df)[0]
        predicted_crop = crop_encoder.inverse_transform([prediction])[0]
        st.success(f"🌾 Recommended Crop: **{predicted_crop}**")

    elif choice == "Fertilizer":
        X_fert = fertilizer_df.drop('Fertilizer Name', axis=1)
        y_fert = fertilizer_df['Fertilizer Name']
        fert_model = RandomForestClassifier()
        fert_model.fit(X_fert, y_fert)

        input_df = pd.DataFrame([[temp, humidity, moisture, int(soil_type), int(crop_type),
                                  nitrogen, potassium, phosphorous]],
                                columns=X_fert.columns)
        prediction = fert_model.predict(input_df)[0]
        predicted_fertilizer = fert_encoder.inverse_transform([prediction])[0]
        st.success(f"🧪 Recommended Fertilizer: **{predicted_fertilizer}**")
