import os
import gzip
import joblib
import pandas as pd
import streamlit as st

# --- Configuration ---------------------------------------------------------
# The model file `best_model.pkl.gz` lives in the same directory as app.py
# inside the Hugging Face Space container (synced from the GitHub repo by the
# sync-to-space.yml workflow). Load it directly — no Hub download needed.
MODEL_PATH = os.environ.get("MODEL_PATH", "best_model.pkl.gz")

# --- Model loading (cached) ------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {os.path.abspath(MODEL_PATH)}")
        st.info("Make sure `best_model.pkl.gz` is committed to the GitHub repo and synced to this Space.")
        st.stop()
    with gzip.open(MODEL_PATH, "rb") as f:
        return joblib.load(f)

st.set_page_config(page_title="Wellness Tourism Purchase Predictor", page_icon="🏝️")
st.title("🏝️ Wellness Tourism Package — Purchase Predictor")
st.markdown("Enter the prospective customer's details in the sidebar and click **Predict**.")

model = load_model()

# --- Sidebar input form ----------------------------------------------------
st.sidebar.header("Customer Details")
age = st.sidebar.slider("Age", 18, 70, 35)
type_of_contact = st.sidebar.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
city_tier = st.sidebar.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.sidebar.slider("Duration of Pitch (minutes)", 5, 60, 15)
occupation = st.sidebar.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
number_of_person_visiting = st.sidebar.slider("Number of Persons Visiting", 1, 5, 3)
number_of_followups = st.sidebar.slider("Number of Follow-ups", 1, 6, 4)
product_pitched = st.sidebar.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
preferred_property_star = st.sidebar.slider("Preferred Property Star", 3, 5, 3)
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
number_of_trips = st.sidebar.slider("Number of Trips (annual)", 1, 20, 3)
passport = st.sidebar.selectbox("Passport", [0, 1])
pitch_satisfaction_score = st.sidebar.slider("Pitch Satisfaction Score", 1, 5, 3)
own_car = st.sidebar.selectbox("Owns Car", [0, 1])
number_of_children_visiting = st.sidebar.slider("Number of Children Visiting", 0, 3, 1)
designation = st.sidebar.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.sidebar.slider("Monthly Income", 10000, 100000, 25000, step=1000)

# --- Build input dataframe matching training schema -----------------------
input_dict = {
    "Age": age,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "NumberOfFollowups": number_of_followups,
    "PreferredPropertyStar": preferred_property_star,
    "NumberOfTrips": number_of_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "MonthlyIncome": monthly_income,
    "TypeofContact_Self Enquiry": 1 if type_of_contact == "Self Enquiry" else 0,
    "Occupation_Large Business": 1 if occupation == "Large Business" else 0,
    "Occupation_Salaried": 1 if occupation == "Salaried" else 0,
    "Occupation_Small Business": 1 if occupation == "Small Business" else 0,
    "Gender_Male": 1 if gender == "Male" else 0,
    "ProductPitched_Deluxe": 1 if product_pitched == "Deluxe" else 0,
    "ProductPitched_King": 1 if product_pitched == "King" else 0,
    "ProductPitched_Standard": 1 if product_pitched == "Standard" else 0,
    "ProductPitched_Super Deluxe": 1 if product_pitched == "Super Deluxe" else 0,
    "MaritalStatus_Married": 1 if marital_status == "Married" else 0,
    "MaritalStatus_Single": 1 if marital_status == "Single" else 0,
    "MaritalStatus_Unmarried": 1 if marital_status == "Unmarried" else 0,
    "Designation_Executive": 1 if designation == "Executive" else 0,
    "Designation_Manager": 1 if designation == "Manager" else 0,
    "Designation_Senior Manager": 1 if designation == "Senior Manager" else 0,
    "Designation_VP": 1 if designation == "VP" else 0,
}

# Align columns with the model's training schema
input_df = pd.DataFrame([input_dict])
for col in model.feature_names_in_:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model.feature_names_in_]

st.subheader("Input Summary")
st.dataframe(input_df.T, height=400)

# --- Prediction ------------------------------------------------------------
if st.sidebar.button("Predict", type="primary"):
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0, 1])
    if prediction == 1:
        st.success(f"✅ Likely to PURCHASE the Wellness Tourism Package (probability = {probability:.1%})")
    else:
        st.warning(f"❌ Unlikely to purchase (probability = {probability:.1%})")
    st.metric("Purchase Probability", f"{probability:.1%}")
