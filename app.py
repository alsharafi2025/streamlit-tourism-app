import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Wellness Tourism Predictor", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

model = load_model()

st.title("Wellness Tourism Package Predictor")
st.write("Enter customer details below to predict whether the customer is likely to purchase the package.")

age = st.number_input("Age", min_value=18, max_value=100, value=35)
typeofcontact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
citytier = st.selectbox("City Tier", [1, 2, 3])
durationofpitch = st.number_input("Duration of Pitch", min_value=1, max_value=60, value=15)
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
numberofpersonvisiting = st.number_input("Number of Person Visiting", min_value=1, max_value=10, value=2)
numberoffollowups = st.number_input("Number of Followups", min_value=0, max_value=10, value=3)
productpitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
preferredpropertystar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
maritalstatus = st.selectbox("Marital Status", ["Married", "Unmarried", "Divorced"])
numberoftrips = st.number_input("Number of Trips", min_value=0, max_value=20, value=2)
passport = st.selectbox("Passport", [0, 1])
pitchsatisfactionscore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
owncar = st.selectbox("Own Car", [0, 1])
numberofchildrenvisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthlyincome = st.number_input("Monthly Income", min_value=1000, max_value=1000000, value=20000)

input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeofcontact,
    "CityTier": citytier,
    "DurationOfPitch": durationofpitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": numberofpersonvisiting,
    "NumberOfFollowups": numberoffollowups,
    "ProductPitched": productpitched,
    "PreferredPropertyStar": preferredpropertystar,
    "MaritalStatus": maritalstatus,
    "NumberOfTrips": numberoftrips,
    "Passport": passport,
    "PitchSatisfactionScore": pitchsatisfactionscore,
    "OwnCar": owncar,
    "NumberOfChildrenVisiting": numberofchildrenvisiting,
    "Designation": designation,
    "MonthlyIncome": monthlyincome
}])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"Likely to Purchase ✅ | Probability: {probability:.2%}")
    else:
        st.warning(f"Not Likely to Purchase ⚠️ | Probability: {probability:.2%}")

    st.subheader("Input Summary")
    st.dataframe(input_df)
