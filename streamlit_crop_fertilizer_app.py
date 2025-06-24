import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load datasets
@st.cache_data
def load_data():
    fert_df = pd.read_csv("data/Fertilizer Prediction.csv")
    crop_df = pd.read_csv("data/Crop_recommendation.csv")
    return fert_df, crop_df

fertilizer_df, crop_recommendation_df = load_data()

# Label encode categorical columns and save encoders for inverse transform
fert_encoder = LabelEncoder()
for col in fertilizer_df.select_dtypes(include=["object"]).columns:
    fertilizer_df[col] = fert_encoder.fit_transform(fertilizer_df[col])

crop_encoder = LabelEncoder()
for col in crop_recommendation_df.select_dtypes(include=["object"]).columns:
    crop_recommendation_df[col] = crop_encoder.fit_transform(crop_recommendation_df[col])

# Train models once
@st.cache_resource
def train_models():
    # Crop Model
    X_crop = crop_recommendation_df.drop('label', axis=1)
    y_crop = crop_recommendation_df['label']
    crop_model = RandomForestClassifier()
    crop_model.fit(X_crop, y_crop)

    # Fertilizer Model
    X_fert = fertilizer_df.drop('Fertilizer Name', axis=1)
    y_fert = fertilizer_df['Fertilizer Name']
    fert_model = RandomForestClassifier()
    fert_model.fit(X_fert, y_fert)

    return crop_model, fert_model, X_crop.columns, X_fert.columns

crop_model, fert_model, crop_features, fert_features = train_models()

# Get unique categories for dropdowns for better UX
soil_types = fertilizer_df['Soil Type'].unique()
crop_types = fertilizer_df['Crop Type'].unique()

# UI Title
st.title("🌾 Crop and Fertilizer Prediction App")

# User choice: Crop or Fertilizer prediction
choice = st.selectbox("What do you want to predict?", ["Crop", "Fertilizer"])

st.subheader("Enter the following parameters:")

# Inputs common for both
temp = st.number_input("🌡️ Temperature")
humidity = st.number_input("💧 Humidity")
moisture = st.number_input("🧪 Moisture")

# Soil and crop type dropdowns (for Fertilizer prediction only)
soil_type_str = st.selectbox("🌱 Soil Type", soil_types)
crop_type_str = st.selectbox("🌾 Crop Type", crop_types)

# Nutrients and other inputs
nitrogen = st.number_input("🧬 Nitrogen")
potassium = st.number_input("🧪 Potassium")
phosphorous = st.number_input("🧪 Phosphorous")
ph = st.number_input("pH")
rainfall = st.number_input("🌧️ Rainfall")

# Encode selected soil and crop types for Fertilizer prediction input
soil_type_encoded = fert_encoder.transform([soil_type_str])[0]
crop_type_encoded = fert_encoder.transform([crop_type_str])[0]

if st.button("🔍 Predict"):
    if choice == "Crop":
        # Prepare input dataframe matching crop model features
        input_df = pd.DataFrame([[nitrogen, phosphorous, potassium, temp, humidity, ph, rainfall]],
                                columns=crop_features)
        prediction = crop_model.predict(input_df)[0]
        predicted_crop = crop_encoder.inverse_transform([prediction])[0]
        st.success(f"🌾 Recommended Crop: **{predicted_crop}**")

    else:  # Fertilizer prediction
        input_df = pd.DataFrame([[temp, humidity, moisture, soil_type_encoded, crop_type_encoded,
                                  nitrogen, potassium, phosphorous]],
                                columns=fert_features)
        prediction = fert_model.predict(input_df)[0]
        predicted_fertilizer = fert_encoder.inverse_transform([prediction])[0]
        st.success(f"🧪 Recommended Fertilizer: **{predicted_fertilizer}**")
