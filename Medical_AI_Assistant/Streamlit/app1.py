import streamlit as st
import joblib
import pandas as pd
import os
import numpy as np

# =========================
# Paths
# =========================
MODEL_PATH = '../models/best_ml_combined_model.pkl'
SYMPTOM_INDEX_PATH = '../notebooks/symptom_index.pkl'
DESCRIPTION_PATH = '../data/Cleaned_Symptom_description.csv'
PRECAUTION_PATH = '../data/Cleaned_Symptom_precautions.csv'
SEVERITY_PATH = '../data/Cleaned_Symptom_severity.csv'
LABEL_ENCODER_PATH = '../models/label_encoder.pkl'


# =========================
# Custom Medical Background (EKG Style)
# =========================
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.65)),
                url("https://tse1.mm.bing.net/th/id/OIP.N6a3y1FZQItHe-Hqtnwt4wHaEI?pid=Api");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: white;
}

[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}

[data-testid="stToolbar"] {
    right: 2rem;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)



# =========================
# Load model & data
# =========================
@st.cache_resource(show_spinner=False)
def load_model_and_index():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        return None, None
    if not os.path.exists(SYMPTOM_INDEX_PATH):
        st.error(f"Symptom index file not found: {SYMPTOM_INDEX_PATH}")
        return None, None
    try:
        model = joblib.load(MODEL_PATH)
        symptom_index = joblib.load(SYMPTOM_INDEX_PATH)
        return model, symptom_index
    except Exception as e:
        st.error(f"Error loading model or index: {e}")
        return None, None

@st.cache_data(show_spinner=False)
def load_data():
    try:
        description = pd.read_csv(DESCRIPTION_PATH)
    except Exception:
        description = pd.DataFrame()
    try:
        precaution = pd.read_csv(PRECAUTION_PATH)
    except Exception:
        precaution = pd.DataFrame()
    try:
        severity = pd.read_csv(SEVERITY_PATH)
    except Exception:
        severity = pd.DataFrame()
    return description, precaution, severity


# =========================
# Helper Functions
# =========================
def prettify(symptom):
    return symptom.replace("_", " ").title()

def normalize_symptom(symptom):
    return symptom.strip().lower().replace(" ", "_")

def get_disease_info(disease, description, precaution):
    """Fetch description, drugs, precautions safely"""
    drugs = []
    precautions = []

    if not description.empty:
        try:
            drugs_arr = description[description["Disease"].str.lower() == disease.lower()]["Description"].values
            if len(drugs_arr) > 0:
                if isinstance(drugs_arr[0], str):
                    drugs = [d.strip() for d in drugs_arr[0].split(",") if d.strip()]
        except Exception:
            drugs = []

    if not precaution.empty:
        try:
            row = precaution[precaution["Disease"].str.lower() == disease.lower()]
            if not row.empty:
                precautions = row.iloc[0, 1:].dropna().tolist()
        except Exception:
            precautions = []

    # ensure plain python list
    def ensure_list(x):
        if isinstance(x, (np.ndarray, pd.Series)):
            return [str(i).strip() for i in x.tolist() if str(i).strip()]
        if isinstance(x, (list, tuple, set)):
            return [str(i).strip() for i in x if str(i).strip()]
        if isinstance(x, str) and x.strip():
            return [x.strip()]
        return []

    drugs = ensure_list(drugs)
    precautions = ensure_list(precautions)

    return drugs if len(drugs) > 0 else ["No drugs found"], precautions if len(precautions) > 0 else ["No precautions found"]

def get_symptom_severity(user_symptoms, severity):
    """Get severity score for each symptom"""
    severities = {}
    if severity.empty:
        return {sym: "Unknown" for sym in user_symptoms}
    for symptom in user_symptoms:
        try:
            match = severity[severity["Symptom"].str.lower() == symptom.lower()]
            if not match.empty:
                severities[symptom] = match["weight"].values[0]
            else:
                severities[symptom] = "Unknown"
        except Exception:
            severities[symptom] = "Unknown"
    return severities


# Load label encoder (ek hi dafa top pe)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

def predict_disease(user_symptoms, model, symptom_index, all_symptoms):
    """Generate prediction vector and predict disease"""
    input_vector = [0] * len(all_symptoms)
    for symptom in user_symptoms:
        if symptom in symptom_index:
            input_vector[symptom_index[symptom]] = 1
    try:
        prediction = model.predict([input_vector])[0]
        predicted_disease = label_encoder.inverse_transform([prediction])[0]
        return predicted_disease
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


# =========================
# UI Design
# =========================
st.set_page_config(page_title="Medical AI Assistant", page_icon="🧑‍⚕️", layout="wide")

st.title("🩺 Medical AI Disease Predictor")
st.markdown("Enter your symptoms below **or** pick from the list.")

# Load resources
model, symptom_index = load_model_and_index()
description, precaution, severity = load_data()

if model is not None and symptom_index is not None:
    all_symptoms = list(symptom_index.keys())
    display_options = [prettify(s) for s in all_symptoms]
    display_to_token = {prettify(s): s for s in all_symptoms}

    tab_pick, tab_type = st.tabs(["Pick from list", "Type manually"])

    with tab_pick:
        picked_display = st.multiselect(
            "Select symptoms (searchable)",
            options=display_options
        )
        picked_tokens = [display_to_token[x] for x in picked_display]

    with tab_type:
        user_input = st.text_input("💡 Symptoms (comma separated)", "")
        typed_tokens = [normalize_symptom(sym) for sym in user_input.split(",") if sym.strip()]

    # merge inputs
    user_symptoms = sorted(set(picked_tokens + typed_tokens))

    if st.button("🔍 Diagnose", use_container_width=True):
        if not user_symptoms:
            st.warning("⚠️ Please select or type at least one symptom.")
        else:
            invalid = [s for s in user_symptoms if s not in all_symptoms]
            if invalid:
                st.error(f"❌ Invalid symptoms: {invalid}")
                with st.expander("📋 See all available symptoms"):
                    st.write(sorted(all_symptoms))
            else:
                disease = predict_disease(user_symptoms, model, symptom_index, all_symptoms)

                if disease:
                    st.success(f"✅ Predicted Disease: **{disease}**")

                    drugs, precautions = get_disease_info(disease, description, precaution)
                    severities = get_symptom_severity(user_symptoms, severity)

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.subheader("💊 Drugs / Description")
                        for d in drugs:
                            st.write(f"- {d}")

                    with col2:
                        st.subheader("🛡️ Precautions")
                        for p in precautions:
                            st.write(f"- {p}")

                    with col3:
                        st.subheader("📊 Symptom Severity")
                        for s, sev in severities.items():
                            st.write(f"- **{prettify(s)}** → {sev}")

                else:
                    st.error("Could not predict disease. Please check your model and input.")
    else:
        st.info("ℹ️ Select or type your symptoms above and click Diagnose.")
else:
    st.stop()
