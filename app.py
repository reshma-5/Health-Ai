import streamlit as st
import requests

# WatsonX credentials
api_key = st.secrets["WATSONX_API_KEY"]
project_id = st.secrets["WATSONX_PROJECT_ID"]
model_id = "granite-3b-instruct-v1"

# IAM token function
@st.cache_resource
def get_iam_token():
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = f"grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={api_key}"
    response = requests.post(url, headers=headers, data=data)
    return response.json()["access_token"]

# Function to query Granite
def query_granite(prompt):
    token = get_iam_token()
    url = "https://us-south.ml.cloud.ibm.com/v2/inference"  # ✅ your region is us-south (Dallas)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "model_id": model_id,
        "input": prompt,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 300
        },
        "project_id": project_id 
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        try:
            return response.json()["results"][0]["generated_text"]
        except (KeyError, IndexError):
            return "⚠️ Model responded but no text was returned."
    else:
        return f"❌ Error: {response.status_code} - {response.text}"


# Streamlit UI
st.set_page_config(page_title="HealthAI", page_icon="🩺", layout="centered")
st.sidebar.title("🩺 HealthAI Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🗣️ Patient Chat", "🔍 Disease Prediction", "💊 Treatment Plan"])

if page == "🏠 Home":
    st.title("🏠 Welcome to HealthAI")
    st.markdown("Powered by IBM WatsonX + Granite model")

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
