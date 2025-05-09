import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv
import time
import httpx
from streamlit_lottie import st_lottie
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from PIL import Image
import base64

# Load environment variables
load_dotenv()

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
DEFAULT_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Page configuration
st.set_page_config(
    page_title="AI Story & Translation Hub",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        background-image: linear-gradient(to bottom right, #f5f7f9, #e8eef2);
    }
    .stTextInput, .stSelectbox, .stTextarea {
        background-color: #ffffff;
        border-radius: 10px;
        border: 1px solid #e0e4e8;
        padding: 10px;
    }
    .stButton>button {
        background-color: #7E57C2;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #5E35B1;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .output-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        margin-top: 20px;
    }
    h1, h2, h3 {
        color: #512DA8;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 6px 6px 0px 0px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #7E57C2;
        color: white;
    }
    .error-message {
        color: #D32F2F;
        background-color: #FFEBEE;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #D32F2F;
    }
    .success-message {
        color: #2E7D32;
        background-color: #E8F5E9;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #2E7D32;
    }
    .api-check {
        background-color: #E8F5E9;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .model-selection {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .gradient-text {
        background: linear-gradient(45deg, #512DA8, #7E57C2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .api-status-container {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 5px;
    }
    .status-indicator {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }
    .status-online {
        background-color: #4CAF50;
    }
    .status-offline {
        background-color: #F44336;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def load_lottie_animation(url):
    try:
        with httpx.Client() as client:
            response = client.get(url)
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        st.error(f"Error loading animation: {e}")
    return None

def verify_api_key(api_key):
    """Verify if the API key is valid by checking the health endpoint of the API"""
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.get(f"{API_URL}/health", headers=headers, timeout=10)
        
        if response.status_code == 200:
            return True, "API connection successful"
        else:
            return False, f"API Error: {response.status_code} - {response.text}"
    except requests.RequestException as e:
        return False, f"Connection Error: {str(e)}"

def check_api_health():
    """Check if the API is running and accessible"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def generate_story(title, api_key, model):
    """Generate a story using the API"""
    try:
        headers = {
            "Content-Type": "application/json",
            "X-API-KEY": api_key
        }
        data = {"title": title, "model": model}
        response = requests.post(f"{API_URL}/generate", json=data, headers=headers)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            error_detail = "Unknown error"
            try:
                error_detail = response.json().get("detail", "Unknown error")
            except:
                pass
            return False, f"Error {response.status_code}: {error_detail}"
    except requests.RequestException as e:
        return False, f"Request failed: {str(e)}"

def summarize_text(text, api_key, model):
    """Summarize text using the API"""
    try:
        headers = {
            "Content-Type": "application/json",
            "X-API-KEY": api_key
        }
        data = {"title": text, "model": model}
        response = requests.post(f"{API_URL}/summarize", json=data, headers=headers)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            error_detail = "Unknown error"
            try:
                error_detail = response.json().get("detail", "Unknown error")
            except:
                pass
            return False, f"Error {response.status_code}: {error_detail}"
    except requests.RequestException as e:
        return False, f"Request failed: {str(e)}"

def translate_text(text, target_language, api_key, model):
    """Translate text using the API"""
    try:
        headers = {
            "Content-Type": "application/json",
            "X-API-KEY": api_key
        }
        data = {"text": text, "target_language": target_language, "model": model}
        response = requests.post(f"{API_URL}/translate", json=data, headers=headers)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            error_detail = "Unknown error"
            try:
                error_detail = response.json().get("detail", "Unknown error")
            except:
                pass
            return False, f"Error {response.status_code}: {error_detail}"
    except requests.RequestException as e:
        return False, f"Request failed: {str(e)}"

# Main app function
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("<h2 class='gradient-text'>✨ AI Story & Translation Hub</h2>", unsafe_allow_html=True)
        
        # API Connection Status
        api_status = check_api_health()
        st.markdown(
            f"""
            <div class="api-status-container">
                <span>API Status:</span>
                <span class="status-indicator {'status-online' if api_status else 'status-offline'}"></span>
                <span>{'Online' if api_status else 'Offline'}</span>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # API Key Setup
        st.markdown("### 🔑 API Key Setup")
        api_key = st.text_input("Enter your Gemini API Key", value=DEFAULT_API_KEY, type="password")
        
        if api_key:
            is_valid, message = verify_api_key(api_key)
            if is_valid:
                st.markdown(f"<div class='success-message'>{message}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='error-message'>{message}</div>", unsafe_allow_html=True)
        
        # Model Selection
        st.markdown("### 🤖 Model Selection")
        model_options = {
            "gemini-1.5-flash": "Gemini 1.5 Flash (Fast)",
            "gemini-1.5-pro": "Gemini 1.5 Pro (Balanced)",
            "gemini-1.5-ultra": "Gemini 1.5 Ultra (Powerful)"
        }
        
        selected_model = st.selectbox(
            "Choose AI Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x]
        )
        
        # Display model details
        model_descriptions = {
            "gemini-1.5-flash": "Fastest model for basic tasks with lower token limit",
            "gemini-1.5-pro": "Balanced performance with medium token limit",
            "gemini-1.5-ultra": "Most powerful model with highest token limit"
        }
        
        st.info(model_descriptions[selected_model])
        
        # Lottie animation in sidebar
        lottie_url = "https://assets5.lottiefiles.com/packages/lf20_kk62um9v.json"
        lottie_anim = load_lottie_animation(lottie_url)
        if lottie_anim:
            st_lottie(lottie_anim, height=200, key="sidebar_animation")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This application uses Google's Gemini AI to generate stories, 
        summarize text, and translate content between languages.
        """)
        st.markdown("Built with Streamlit & FastAPI")
    
    # Main content
    st.markdown("<h1 class='gradient-text'>AI Story & Translation Hub ✨</h1>", unsafe_allow_html=True)
    st.markdown("Generate creative stories, summarize long text, and translate between languages - all powered by AI!")
    
    # Check if API key exists
    if not api_key:
        st.warning("Please enter your Gemini API Key in the sidebar to get started.")
        return
    
    # Tabs
    tabs = st.tabs(["📝 Story Generator", "📚 Text Summarizer", "🌐 Translator"])
    
    # Story Generator Tab
    with tabs[0]:
        st.markdown("### 📝 Generate Creative Stories")
        st.markdown("Enter a title or topic, and let the AI craft a unique story for you!")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            story_title = st.text_input("Story Title or Topic", placeholder="Enter a title or topic for your story")
        
        with col2:
            generate_button = st.button("Generate Story 🚀", use_container_width=True)
        
        if story_title and generate_button:
            with st.spinner("Generating your story..."):
                success, result = generate_story(story_title, api_key, selected_model)
                
                if success:
                    st.markdown(f"<div class='output-container'><h3>{result['title']}</h3>{result['story']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='error-message'>{result}</div>", unsafe_allow_html=True)
    
    # Text Summarizer Tab
    with tabs[1]:
        st.markdown("### 📚 Summarize Long Text")
        st.markdown("Paste your long text below, and get a concise summary!")
        
        text_to_summarize = st.text_area("Text to Summarize", height=200, placeholder="Paste your long text here (minimum 50 characters)")
        
        summarize_button = st.button("Summarize Text 📝", use_container_width=True)
        
        if text_to_summarize and summarize_button:
            if len(text_to_summarize) < 50:
                st.markdown(f"<div class='error-message'>Text is too short. Please provide at least 50 characters for a meaningful summary.</div>", unsafe_allow_html=True)
            else:
                with st.spinner("Summarizing your text..."):
                    success, result = summarize_text(text_to_summarize, api_key, selected_model)
                    
                    if success:
                        st.markdown(f"<div class='output-container'><h3>Summary</h3>{result['summary']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='error-message'>{result}</div>", unsafe_allow_html=True)
    
    # Translator Tab
    with tabs[2]:
        st.markdown("### 🌐 Translate Text")
        st.markdown("Enter text and select a target language for translation!")
        
        text_to_translate = st.text_area("Text to Translate", height=150, placeholder="Enter text to translate")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            languages = [
                "Arabic", "Bengali", "Chinese (Simplified)", "Chinese (Traditional)",
                "Dutch", "English", "French", "German", "Greek", "Hindi", "Indonesian",
                "Italian", "Japanese", "Korean", "Portuguese", "Russian", "Spanish",
                "Swahili", "Tamil", "Thai", "Turkish", "Ukrainian", "Vietnamese"
            ]
            
            target_language = st.selectbox("Target Language", languages)
        
        with col2:
            translate_button = st.button("Translate 🌍", use_container_width=True)
        
        if text_to_translate and translate_button:
            if len(text_to_translate) < 1:
                st.markdown(f"<div class='error-message'>Please enter some text to translate.</div>", unsafe_allow_html=True)
            else:
                with st.spinner(f"Translating to {target_language}..."):
                    success, result = translate_text(text_to_translate, target_language, api_key, selected_model)
                    
                    if success:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"<div class='output-container'><h3>Original Text</h3>{result['original_text']}</div>", unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"<div class='output-container'><h3>Translated Text ({target_language})</h3>{result['translated_text']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='error-message'>{result}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()