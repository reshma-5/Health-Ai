import streamlit as st
import requests

# ✅ WatsonX credentials from Streamlit secrets
api_key = st.secrets["WATSONX_API_KEY"]
project_id = st.secrets["WATSONX_PROJECT_ID"]
region = "us-south"  # Change to your region if needed
model_id = "granite-3b-instruct"  # Or granite-3.3b-instruct-v1

# ✅ Get IAM token from IBM
@st.cache_resource
def get_iam_token():
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = f"grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={api_key}"
    response = requests.post(url, headers=headers, data=data)
    return response.json()["access_token"]

# ✅ Function to query Granite model via WatsonX
def query_granite(prompt):
    token = get_iam_token()
    url = f"https://{region}.ml.cloud.ibm.com/v2/inference"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "model_id": model_id,
        "input": [{"role": "user", "content": prompt}],
        "parameters": {
            "decoding_method": "sample",
            "max_new_tokens": 300,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9
        },
        "project_id": project_id
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        try:
            return response.json()["results"][0]["generated_text"]
        except Exception as e:
            return f"⚠️ Failed to parse model output: {str(e)}"
    else:
        return f"❌ Error: {response.status_code} - {response.text}"

# ✅ Streamlit UI Setup
st.set_page_config(page_title="HealthAI", page_icon="🩺", layout="centered")
st.sidebar.title("🩺 HealthAI Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🗣️ Patient Chat", "🔍 Disease Prediction", "💊 Treatment Plan"])

if page == "🏠 Home":
    st.title("🏠 Welcome to HealthAI")
    st.markdown("""
        🤖 Ask medical questions  
        🧪 Predict conditions from symptoms  
        💊 Get treatment plans  
        ---
        Powered by **IBM watsonx.ai + Granite**
    """)

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
            reply = query_granite(prompt)
            st.success(reply)

elif page == "💊 Treatment Plan":
    st.title("💊 Treatment Planner")
    condition = st.text_input("Enter diagnosed condition:")
    if condition:
        with st.spinner("Generating plan..."):
            prompt = f"Provide a complete treatment plan for {condition}."
            reply = query_granite(prompt)
            st.success(reply)
