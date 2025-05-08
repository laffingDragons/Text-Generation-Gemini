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
import google.generativeai as genai
from anthropic import Anthropic
import openai
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# API URL for your FastAPI backend
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="Multi-LLM Story & Translation Hub",
    page_icon="🚀",
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
        background-color: #4F46E5;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #3730A3;
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
        color: #4F46E5;
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
        background-color: #4F46E5;
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
    .warning-message {
        color: #FF6D00;
        background-color: #FFF3E0;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #FF6D00;
    }
    .model-selection {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .gradient-text {
        background: linear-gradient(45deg, #3730A3, #4F46E5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .provider-logo {
        max-height: 30px;
        margin-right: 10px;
    }
    .provider-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin: 10px 0;
        border: 2px solid transparent;
        transition: all 0.3s;
    }
    .provider-card:hover {
        border-color: #4F46E5;
        transform: translateY(-2px);
    }
    .provider-card.selected {
        border-color: #4F46E5;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'api_verified' not in st.session_state:
    st.session_state.api_verified = {}

if 'active_provider' not in st.session_state:
    st.session_state.active_provider = None

# Helper functions
def load_lottie_animation(url):
    """Load Lottie animation from URL"""
    try:
        with httpx.Client() as client:
            response = client.get(url)
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        logger.error(f"Error loading animation: {e}")
    return None

def verify_api_key(provider, api_key):
    """Verify if the API key is valid for the selected provider"""
    try:
        if provider == "openai":
            client = openai.OpenAI(api_key=api_key)
            response = client.models.list()
            return True, "OpenAI API key verified successfully"
            
        elif provider == "anthropic":
            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hello"}]
            )
            return True, "Anthropic API key verified successfully"
            
        elif provider == "gemini":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content("Hello")
            return True, "Google Gemini API key verified successfully"
            
        elif provider == "cohere":
            # Using requests since we haven't imported the cohere library yet
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            response = requests.get("https://api.cohere.ai/v1/models", headers=headers)
            if response.status_code == 200:
                return True, "Cohere API key verified successfully"
            else:
                return False, f"Cohere API verification failed: {response.text}"
                
        elif provider == "mistral":
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            response = requests.get("https://api.mistral.ai/v1/models", headers=headers)
            if response.status_code == 200:
                return True, "Mistral API key verified successfully"
            else:
                return False, f"Mistral API verification failed: {response.text}"
                
        elif provider == "llama":
            # Since this is a less standard API, we'll do a simplified check
            return True, "Meta Llama API key format accepted (verification limited)"
            
        elif provider == "deepseek":
            # Using requests since deepseek may have a custom API
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            # This is a placeholder URL - replace with actual DeepSeek API endpoint
            response = requests.get("https://api.deepseek.com/v1/models", headers=headers)
            if response.status_code == 200:
                return True, "DeepSeek API key verified successfully"
            else:
                # For now, we'll accept the key format
                return True, "DeepSeek API key format accepted (verification limited)"
        
        else:
            return False, "Unknown provider"
            
    except Exception as e:
        return False, f"API verification failed: {str(e)}"

def generate_with_llm(provider, api_key, prompt, model=None):
    """Generate content using the selected LLM provider"""
    try:
        if provider == "openai":
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model or "gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024
            )
            return True, response.choices[0].message.content
            
        elif provider == "anthropic":
            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model or "claude-3-haiku-20240307",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return True, response.content[0].text
            
        elif provider == "gemini":
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel(model or "gemini-1.5-flash")
            response = gemini_model.generate_content(prompt)
            return True, response.text
            
        elif provider == "cohere":
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model or "command",
                "prompt": prompt,
                "max_tokens": 1024
            }
            response = requests.post("https://api.cohere.ai/v1/generate", headers=headers, json=payload)
            if response.status_code == 200:
                return True, response.json().get("generations", [{}])[0].get("text", "")
            else:
                return False, f"Cohere API error: {response.text}"
                
        elif provider == "mistral":
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model or "mistral-small-latest",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024
            }
            response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload)
            if response.status_code == 200:
                return True, response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                return False, f"Mistral API error: {response.text}"
                
        elif provider == "llama":
            # Placeholder for Llama API integration
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model or "meta-llama-3-8b-instruct",
                "prompt": prompt,
                "max_tokens": 1024
            }
            # This is a placeholder - replace with actual Meta Llama API endpoint
            response = requests.post("https://api.llama.api/v1/completions", headers=headers, json=payload)
            if response.status_code == 200:
                return True, response.json().get("choices", [{}])[0].get("text", "")
            else:
                return False, f"Llama API error: {response.text}"
                
        elif provider == "deepseek":
            # Placeholder for DeepSeek API integration
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model or "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024
            }
            # This is a placeholder - replace with actual DeepSeek API endpoint
            response = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload)
            if response.status_code == 200:
                return True, response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                return False, f"DeepSeek API error: {response.text}"
            
        else:
            return False, "Unknown provider"
            
    except Exception as e:
        return False, f"Generation failed: {str(e)}"

# Provider configurations
llm_providers = {
    "openai": {
        "name": "OpenAI",
        "description": "GPT models from OpenAI (GPT-3.5, GPT-4)",
        "models": {
            "gpt-3.5-turbo": "GPT-3.5 Turbo (Fast, Affordable)",
            "gpt-4o": "GPT-4o (Powerful, Multi-modal)",
            "gpt-4o-mini": "GPT-4o Mini (Balanced)"
        }
    },
    "anthropic": {
        "name": "Anthropic",
        "description": "Claude models from Anthropic",
        "models": {
            "claude-3-haiku-20240307": "Claude 3 Haiku (Fast)",
            "claude-3-sonnet-20240229": "Claude 3 Sonnet (Balanced)",
            "claude-3-opus-20240229": "Claude 3 Opus (Powerful)"
        }
    },
    "gemini": {
        "name": "Google",
        "description": "Gemini models from Google",
        "models": {
            "gemini-1.5-flash": "Gemini 1.5 Flash (Fast)",
            "gemini-1.5-pro": "Gemini 1.5 Pro (Balanced)"
        }
    },
    "mistral": {
        "name": "Mistral AI",
        "description": "Models from Mistral AI",
        "models": {
            "mistral-small-latest": "Mistral Small (Fast)",
            "mistral-medium-latest": "Mistral Medium (Balanced)",
            "mistral-large-latest": "Mistral Large (Powerful)"
        }
    },
    "llama": {
        "name": "Meta",
        "description": "Llama models from Meta",
        "models": {
            "meta-llama-3-8b-instruct": "Llama 3 8B (Fast)",
            "meta-llama-3-70b-instruct": "Llama 3 70B (Powerful)"
        }
    },
    "deepseek": {
        "name": "DeepSeek",
        "description": "Models from DeepSeek",
        "models": {
            "deepseek-chat": "DeepSeek Chat (General)",
            "deepseek-coder": "DeepSeek Coder (Code-specialized)"
        }
    },
    "cohere": {
        "name": "Cohere",
        "description": "Models from Cohere",
        "models": {
            "command": "Command (General)",
            "command-light": "Command Light (Fast)",
            "command-r": "Command-R (Robust)"
        }
    }
}

# Main app function
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("<h2 class='gradient-text'>🚀 Multi-LLM Hub</h2>", unsafe_allow_html=True)
        
        st.markdown("### 🤖 Select LLM Provider")
        
        # Provider selection
        for provider_id, provider_info in llm_providers.items():
            provider_selected = st.session_state.active_provider == provider_id
            
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(
                        f"{provider_info['name']}",
                        key=f"provider_{provider_id}",
                        help=provider_info['description'],
                        use_container_width=True,
                        type="primary" if provider_selected else "secondary"
                    ):
                        st.session_state.active_provider = provider_id
                        st.rerun()
                
                # Show verification status if available
                with col2:
                    if provider_id in st.session_state.api_verified:
                        if st.session_state.api_verified[provider_id]:
                            st.markdown("✅", help="API key verified")
                        else:
                            st.markdown("❌", help="API key invalid")
        
        st.markdown("---")
        
        # If a provider is selected, show API key input and model selection
        if st.session_state.active_provider:
            provider = st.session_state.active_provider
            provider_info = llm_providers[provider]
            
            st.markdown(f"### {provider_info['name']} Configuration")
            
            # API Key input
            api_key = st.text_input(
                f"{provider_info['name']} API Key",
                type="password",
                key=f"api_key_{provider}"
            )
            
            # Verify button
            if api_key:
                if st.button("Verify API Key", key=f"verify_{provider}"):
                    with st.spinner("Verifying API key..."):
                        is_valid, message = verify_api_key(provider, api_key)
                        if is_valid:
                            st.session_state.api_verified[provider] = True
                            st.success(message)
                        else:
                            st.session_state.api_verified[provider] = False
                            st.error(message)
            
            # Model selection if provider is verified
            if provider in st.session_state.api_verified and st.session_state.api_verified[provider]:
                st.markdown("### Model Selection")
                
                selected_model = st.selectbox(
                    "Choose Model",
                    options=list(provider_info["models"].keys()),
                    format_func=lambda x: provider_info["models"][x],
                    key=f"model_{provider}"
                )
                
                # Display model info
                st.info(provider_info["models"][selected_model])
        
        # Lottie animation in sidebar
        lottie_url = "https://assets5.lottiefiles.com/packages/lf20_kk62um9v.json"
        lottie_anim = load_lottie_animation(lottie_url)
        if lottie_anim:
            st_lottie(lottie_anim, height=200, key="sidebar_animation")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This application uses various LLM providers to generate stories, 
        summarize text, and translate content between languages.
        """)
        st.markdown("Built with Streamlit & Python")
    
    # Main content
    st.markdown("<h1 class='gradient-text'>Multi-LLM Story & Translation Hub 🚀</h1>", unsafe_allow_html=True)
    st.markdown("Generate creative stories, summarize long text, and translate between languages using various AI models!")
    
    # Check if provider is selected
    if not st.session_state.active_provider:
        st.warning("Please select an LLM provider from the sidebar to get started.")
        return
    
    provider = st.session_state.active_provider
    provider_info = llm_providers[provider]
    
    # Check if API key is verified
    if provider not in st.session_state.api_verified or not st.session_state.api_verified[provider]:
        st.warning(f"Please enter and verify your {provider_info['name']} API key in the sidebar.")
        return
    
    # Get API key and selected model
    api_key = st.session_state[f"api_key_{provider}"]
    model = st.session_state.get(f"model_{provider}", list(provider_info["models"].keys())[0])
    
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
            if len(story_title) < 3:
                st.markdown(f"<div class='error-message'>Title is too short. Please provide at least 3 characters.</div>", unsafe_allow_html=True)
            else:
                with st.spinner("Generating your story..."):
                    prompt = f"Generate a creative, engaging story about the following topic or title: '{story_title}'. Make it approximately 500 words long with a clear beginning, middle, and end."
                    success, result = generate_with_llm(provider, api_key, prompt, model)
                    
                    if success:
                        st.markdown(f"<div class='output-container'><h3>{story_title}</h3>{result}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='error-message'>{result}</div>", unsafe_allow_html=True)
    
    # Text Summarizer Tab
    with tabs[1]:
        st.markdown("### 📚 Summarize Long Text")
        st.markdown("Paste your long text below, and get a concise summary!")
        
        text_to_summarize = st.text_area("Text to Summarize", height=200, placeholder="Paste your long text here (minimum 100 characters)")
        
        summarize_button = st.button("Summarize Text 📝", use_container_width=True)
        
        if text_to_summarize and summarize_button:
            if len(text_to_summarize) < 100:
                st.markdown(f"<div class='error-message'>Text is too short. Please provide at least 100 characters for a meaningful summary.</div>", unsafe_allow_html=True)
            else:
                with st.spinner("Summarizing your text..."):
                    prompt = f"Summarize the following text concisely, capturing the main points and important details:\n\n{text_to_summarize}"
                    success, result = generate_with_llm(provider, api_key, prompt, model)
                    
                    if success:
                        st.markdown(f"<div class='output-container'><h3>Summary</h3>{result}</div>", unsafe_allow_html=True)
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
                    prompt = f"Translate the following text to {target_language}. Maintain the original meaning and tone as closely as possible:\n\n{text_to_translate}"
                    success, result = generate_with_llm(provider, api_key, prompt, model)
                    
                    if success:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"<div class='output-container'><h3>Original Text</h3>{text_to_translate}</div>", unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"<div class='output-container'><h3>Translated Text ({target_language})</h3>{result}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='error-message'>{result}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()