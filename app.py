import streamlit as st
import requests

# WatsonX credentials (stored in Streamlit secrets)
api_key = st.secrets["WATSONX_API_KEY"]
project_id = st.secrets["WATSONX_PROJECT_ID"]
model_id = "granite-3b-instruct"  # ✅ Use official model ID (no "ibm/" prefix)
region = "us-south"  # ✅ Your region (Dallas)

# Get IAM token from IBM Cloud
@st.cache_resource
def get_iam_token():
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = f"grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={api_key}"
    response = requests.post(url, headers=headers, data=data)
    return response.json()["access_token"]

# Query Granite model via WatsonX
def query_granite(prompt):
    token = get_iam_token()
    url = f"https://{region}.ml.cloud.ibm.com/ml/v1/text-generation?version=2024-05-01"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "model_id": model_id,
        "project_id": project_id,
        "input": prompt,
        "parameters": {
            "decoding_method": "sample",
            "max_new_tokens": 300,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95
        }
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        try:
            return response.json()["results"][0]["generated_text"]
        except (KeyError, IndexError):
            return "⚠️ Model responded, but no generated text was found."
    else:
        return f"❌ Error: {response.status_code} - {response.text}"

# Streamlit UI
st.set_page_config(page_title="HealthAI", page_icon="🩺", layout="centered")
st.sidebar.title("🩺 HealthAI Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🗣️ Patient Chat", "🔍 Disease Prediction", "💊 Treatment Plan"])

if page == "🏠 Home":
    st.title("🏠 Welcome to HealthAI")
    st.markdown("🔹 Ask medical questions\n🔹 Predict diseases\n🔹 Get treatment plans\n\nPowered by **IBM watsonx.ai + Granite**.")

elif page == "🗣️ Patient Chat":
    st.title("🧠 Patient Chat")
    q = st.text_input("Ask your medical question:")
    if q:
        with st.spinner("Thinking..."):
            prompt = f"You are a healthcare assistant. Help the patient:\n{q}"
            reply = query_granite(prompt)
            st.success(reply)

elif page == "🔍 Disease Prediction":
    st.title("🔍 Disease Predictor")
    symptoms = st.text_area("List your symptoms:")
    if symptoms:
        with st.spinner("Analyzing..."):
            prompt = f"A patient reports: {symptoms}. Suggest possible conditions and actions."
            st.success(query_granite(prompt))

elif page == "💊 Treatment Plan":
    st.title("💊 Treatment Planner")
    condition = st.text_input("Enter diagnosed condition:")
    if condition:
        with st.spinner("Generating plan..."):
            prompt = f"Provide a complete treatment plan for {condition}."
            st.success(query_granite(prompt))
