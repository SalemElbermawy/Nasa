# app.py
import streamlit as st
import google.generativeai as gen_ai


GOOGLE_API_KEY = "AIzaSyD5zrNtvWz839V914eyOGQIRrZH8fAGFGA"
LIVEKIT_URL = "wss://mia-s0oyhmq9.livekit.cloud"
LIVEKIT_API_KEY = "APImaMo7Fie6kDA"
LIVEKIT_API_SECRET = "u6Ql18713wmUNsI2Xs2UKLfhVUrMadwjmTwFYi3yy5I"
VITE_LIVEKIT_URL = "wss://mia-s0oyhmq9.livekit.cloud"
GOOGLE_APPLICATION_CREDENTIALS = "D:/project/Mia3/miaa-464511-94eb4bbd3ba0.json"
OPENROUTER_API_KEY = "sk-or-v1-6e64be4b41c01f0a84cff8b16cf373c40fa0d2363e44331fedcdd84650d8702"
OPENROUTER_MODEL = "DeepSeek-R1"
GOOGLE_CSE_ID = "c308c47ea521b4ce3"
IMGBB_API_KEY = "ef1981a17bc6a1ab45a03a01e1db03c3"
SOCCERSAPI_USER = "ziademad17z7"
SOCCERSAPI_TOKEN = "97eb5e809d846a0ee9cf01d10423cd19"
OPENAI_API_KEY = "sk-or-v1-6e64be4b41c01f0a84cff8b16cf373c40fa0d2363e44331fedcdd84650d8702"
SERPAPI_KEY = "e0af34c9ff9fe61cb8e4f388868bbfde9ac84b1e4651ab8d94896ba2decc51fc"
PERSONA_ID = "p63abee1ca9f"
REPLICA_ID = "rf4703150052"
TAVUS_API_KEY = "8ace1e87466b40b08c28c2a3b756c333"
SERPER_API_KEY = "d58de4f56084574c62f2e89011b5b1e9b8ab0d02"


st.set_page_config(
    page_title="NASA Chatbot - AI Earth Lens",
    page_icon="🚀",
    layout="centered",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0e2a 0%, #1a1f4b 50%, #0a0e2a 100%);
        color: white;
    }
    
    /* Header Styles */
    .main-header { 
        text-align: center; 
        margin-bottom: 30px; 
        padding: 20px; 
        background: rgba(255, 255, 255, 0.95); 
        border-radius: 15px; 
        border: 1px solid rgba(124, 240, 61, 0.3);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 { 
        font-size: 2.5em; 
        margin-bottom: 10px; 
        background: linear-gradient(45deg, #7cf03d, #3df0a1); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        background-clip: text; 
        text-shadow: 0 0 20px rgba(124, 240, 61, 0.5);
    }
    .main-header p { 
        font-size: 1.1em; 
        color: #333;
        font-weight: 500;
    }
    
    /* Sidebar Styles - White Background */
    [data-testid="stSidebar"] {
        background-color: white !important;
    }
    .sidebar-content { 
        color: #333 !important; 
        padding: 15px;
    }
    .sidebar-content h2 { 
        color: #7cf03d !important; 
        border-bottom: 2px solid #7cf03d; 
        padding-bottom: 10px;
        font-weight: bold;
    }
    .sidebar-content p, .sidebar-content li { 
        color: #333 !important; 
        font-size: 14px;
    }
    .sidebar-content ul {
        color: #333;
    }
    .sidebar-content li {
        margin-bottom: 8px;
        color: #444;
    }
    
    /* Chat Message Styles */
    .user-msg { 
        background: linear-gradient(135deg, #7cf03d, #3df0a1); 
        color: black; 
        padding: 12px 18px; 
        border-radius: 18px 18px 4px 18px; 
        margin: 8px 0; 
        max-width: 75%; 
        box-shadow: 0 4px 12px rgba(124, 240, 61, 0.3); 
        float: right; 
        clear: both; 
        position: relative; 
        font-size: 14px; 
        line-height: 1.4; 
        font-weight: 500;
    }
    .assistant-msg { 
        background: linear-gradient(135deg, #2c5364, #203a43); 
        color: white; 
        padding: 12px 18px; 
        border-radius: 18px 18px 18px 4px; 
        margin: 8px 0; 
        max-width: 75%; 
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2); 
        float: left; 
        clear: both; 
        position: relative; 
        font-size: 14px; 
        line-height: 1.4;
    }
    
    /* Chat Container */
    .stChat { 
        background: rgba(255, 255, 255, 0.1); 
        backdrop-filter: blur(10px); 
        border-radius: 15px; 
        padding: 20px; 
        margin: 10px 0; 
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Input Styles */
    .stChatInput { 
        background: rgba(255, 255, 255, 0.15) !important; 
        border-radius: 25px !important; 
        border: 1px solid rgba(124, 240, 61, 0.3) !important; 
        color: white !important;
    }
    .stChatInput input {
        color: white !important;
    }
    .stChatInput input::placeholder {
        color: rgba(255, 255, 255, 0.7) !important;
    }
    
    /* Button Styles */
    .stButton>button { 
        background: linear-gradient(45deg, #7cf03d, #3df0a1) !important; 
        color: black !important; 
        border: none !important; 
        border-radius: 25px !important; 
        padding: 10px 20px !important; 
        font-weight: bold !important; 
        width: 100% !important; 
        transition: all 0.3s !important;
    }
    .stButton>button:hover { 
        transform: scale(1.05) !important; 
        box-shadow: 0 5px 15px rgba(124, 240, 61, 0.4) !important;
    }
    
    /* Footer Styles */
    .footer { 
        text-align: center; 
        color: white; 
        opacity: 0.9; 
        margin-top: 30px;
        padding: 15px;
        background: rgba(0, 0, 0, 0.3);
        border-radius: 10px;
    }
    .footer a { 
        color: #7cf03d !important; 
        text-decoration: none;
        font-weight: bold;
    }
    .footer p {
        margin: 5px 0;
        color: white;
    }
    
    /* Ensure text visibility in sidebar */
    [data-testid="stSidebar"] * {
        color: #333 !important;
    }
    
    /* Fix for any white text in sidebar */
    [data-testid="stSidebar"] .css-1d391kg,
    [data-testid="stSidebar"] .css-12oz5g7 {
        color: #333 !important;
    }
</style>
""", unsafe_allow_html=True)


try:
    gen_ai.configure(api_key=GOOGLE_API_KEY)
    model = gen_ai.GenerativeModel("gemini-2.0-flash")
except Exception as e:
    st.error(f"❌ Error configuring Google AI: {e}")
    st.stop()


def map_role(role):
    return "assistant" if role == "model" else role

system_prompt = """
You are a NASA expert assistant with comprehensive knowledge about NASA programs, missions, spacecraft, science, research, technology, history, and education. 
ONLY answer questions about NASA and related STEM fields.
"""


if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])
    st.session_state.chat_session.history = [
        {"role": "user", "parts": [system_prompt]},
        {"role": "model", "parts": ["Hello! I'm your NASA expert assistant. Ask me anything about space missions, astronomy, or NASA! 🚀"]}
    ]


st.markdown("""
<div class="main-header">
    <h1>🚀 NASA Chatbot</h1>
    <p>Your Expert Guide to Space Exploration, Missions, and Astronomy</p>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("""
    <div class="sidebar-content">
        <h2>🌌 About This Chatbot</h2>
        <p><b>Ask me about:</b></p>
        <ul>
            <li>NASA missions & spacecraft</li>
            <li>Astronomy & astrophysics</li>
            <li>Space technology</li>
            <li>NASA history</li>
            <li>Current space research</li>
            <li>Planetary science</li>
        </ul>
        <p><b>Examples:</b></p>
        <ul>
            <li>"Tell me about the Artemis program"</li>
            <li>"What is the James Webb Telescope?"</li>
            <li>"Explain black holes"</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_session.history[2:]:
        role = map_role(message.role)
        text = message.parts[0].text
        if role == "user":
            st.markdown(f"<div class='user-msg'>{text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-msg'>{text}</div>", unsafe_allow_html=True)


user_input = st.chat_input("Ask about NASA missions, space exploration, astronomy...")

if user_input:
    with chat_container:
        st.markdown(f"<div class='user-msg'>{user_input}</div>", unsafe_allow_html=True)
    
    try:
        with st.spinner("🌌 Searching the cosmos..."):
            response = st.session_state.chat_session.send_message(user_input)
        with chat_container:
            st.markdown(f"<div class='assistant-msg'>{response.text}</div>", unsafe_allow_html=True)
    except Exception as e:
        with chat_container:
            st.markdown("<div class='assistant-msg'>🚫 Error processing your question. Please try again.</div>", unsafe_allow_html=True)
        st.error(f"Error: {e}")


st.markdown("""
<div class="footer">
    <p>🚀 NASA Chatbot • Powered by Google Gemini • Part of AI Earth Lens Project</p>
    <p>🌎 Learn more at <a href='https://www.nasa.gov' target='_blank'>nasa.gov</a></p>
</div>
""", unsafe_allow_html=True)
