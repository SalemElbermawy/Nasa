# welcome.py
import streamlit as st
import base64
from pathlib import Path
import streamlit.components.v1 as components

st.set_page_config(
    page_title="AI Earth Lens - NASA Space Apps 2025",
    page_icon="🚀",
    layout="wide"
)

# Unified CSS Theme
st.markdown("""
<style>
    /* 🌌 Global App Background + Cursor */
    .stApp {
        background: linear-gradient(135deg, #0a0e2a 0%, #1a1f4b 50%, #0a0e2a 100%);
        color: white;
        cursor: url('/Nasacursor.cur'), auto !important; /* 🚀 NASA custom cursor */
    }

    /* 🖱️ Pointer for clickable elements (buttons, links, etc.) */
    button, a, .stButton>button {
        cursor: url('/Nasapointer.cur'), pointer !important;
    }

    /* --- Keep your existing styles --- */
    .main-header {
        text-align: center;
        color: white;
        margin-bottom: 30px;
        padding: 20px;
        background: rgba(0, 0, 0, 0.3);
        border-radius: 15px;
        border: 1px solid rgba(124, 240, 61, 0.3);
    }

    .main-header h1 {
        font-size: 3em;
        margin-bottom: 10px;
        background: linear-gradient(45deg, #7cf03d, #3df0a1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 20px rgba(124, 240, 61, 0.5);
    }
    .stApp {
        background: linear-gradient(135deg, #0a0e2a 0%, #1a1f4b 50%, #0a0e2a 100%);
        color: white;
    }
    
    .main-header {
        text-align: center;
        color: white;
        margin-bottom: 30px;
        padding: 20px;
        background: rgba(0, 0, 0, 0.3);
        border-radius: 15px;
        border: 1px solid rgba(124, 240, 61, 0.3);
    }
    
    .main-header h1 {
        font-size: 3em;
        margin-bottom: 10px;
        background: linear-gradient(45deg, #7cf03d, #3df0a1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 20px rgba(124, 240, 61, 0.5);
    }
    
    .main-header h3 {
        font-size: 1.5em;
        opacity: 0.9;
        color: #ccc;
    }
    
    .section-header {
        color: #7cf03d;
        border-bottom: 2px solid #7cf03d;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    .team-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(124, 240, 61, 0.2);
        height: 100%;
    }
    
    .team-card img {
        border-radius: 50%;
        border: 3px solid #7cf03d;
        margin-bottom: 15px;
    }
    
    .info-box {
        background: rgba(124, 240, 61, 0.1);
        border: 1px solid rgba(124, 240, 61, 0.3);
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
    }
    
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #7cf03d, transparent);
        margin: 30px 0;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

def load_image_as_base64(image_path):
    """Load image as base64 string."""
    p = Path(image_path)
    if not p.exists():
        return None
    return base64.b64encode(p.read_bytes()).decode()

def img_to_data_uri(image_path):
    """Convert image to data URI (usable in HTML/CSS)."""
    b64 = load_image_as_base64(image_path)
    if b64:
        return f"data:image/png;base64,{b64}"
    return ""

# Load planet images
sun_img = img_to_data_uri("images/sun.png")
mercury_img = img_to_data_uri("images/mercury.png")
venus_img = img_to_data_uri("images/venus.png")
earth_img = img_to_data_uri("images/earth.png")
moon_img = img_to_data_uri("images/moon.png")
mars_img = img_to_data_uri("images/mars.png")
jupiter_img = img_to_data_uri("images/jupiter.png")
saturn_img = img_to_data_uri("images/saturn.png")
uranus_img = img_to_data_uri("images/uranus.png")
neptune_img = img_to_data_uri("images/neptune.png")
pluto_img = img_to_data_uri("images/pluto.png")

# Solar System Animation
solar_system_html = f"""
<div class="container">
  <div class="sun"><img src="{sun_img}" alt="sun"></div>
  <div class="mercury"></div>
  <div class="venus"></div>
  <div class="earth"><div class="moon"></div></div>
  <div class="mars"></div>
  <div class="jupiter"></div>
  <div class="saturn"></div>
  <div class="uranus"></div>
  <div class="neptune"></div>
  <div class="pluto"></div>
</div>

<style>
body {{
    margin: 0;
    height: 100vh;
    background-color: transparent;
    overflow: hidden;
}}
.container {{
    font-size: 6px;
    width: 40em;
    height: 40em;
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%,-50%);
    z-index: -1;
}}
img{{height:130%; width:130%;}}
.sun {{
    display:flex;align-items:center;justify-content:center;
    position:absolute;top:15em;left:15em;width:10em;height:10em;
    border-radius:50%;box-shadow:0 0 3em rgb(255,128,0);
}}
.mercury,.venus,.earth,.moon,.mars,.jupiter,.saturn,.uranus,.neptune,.pluto {{
    position:absolute;border-style:solid;border-color:white transparent transparent transparent;
    border-width:0.1em 0.1em 0 0;border-radius:50%;
}}

.mercury{{top:12.5em;left:12.5em;width:15em;height:15em;animation:orbit 5s linear infinite;}}
.venus{{top:10em;left:10em;width:20em;height:20em;animation:orbit 10s linear infinite;}}
.earth{{top:6em;left:6em;width:28em;height:28em;animation:orbit 15s linear infinite;}}
.moon{{top:2em;right:-1em;width:7em;height:7em;animation:orbit 2s linear infinite;}}
.mars{{top:2em;left:2.5em;width:36em;height:36em;animation:orbit 20s linear infinite;}}
.jupiter{{top:-2em;left:-2em;width:45em;height:45em;animation:orbit 30s linear infinite;}}
.saturn{{top:-7em;left:-7em;width:55em;height:55em;animation:orbit 40s linear infinite;}}
.uranus{{top:-12em;left:-12em;width:65em;height:65em;animation:orbit 50s linear infinite;}}
.neptune{{top:-17em;left:-17em;width:75em;height:75em;animation:orbit 60s linear infinite;}}
.pluto{{top:-22em;left:-22em;width:85em;height:85em;animation:orbit 70s linear infinite;}}

.mercury::before,.venus::before,.earth::before,.moon::before,.mars::before,
.jupiter::before,.saturn::before,.uranus::before,.neptune::before,.pluto::before{{
    content:'';position:absolute;border-radius:50%;
}}

.mercury::before{{top:1.5em;right:0.8em;width:2em;height:2em;background:url('{mercury_img}');background-size:cover;}}
.venus::before{{top:2em;right:2em;width:2em;height:2em;background:url('{venus_img}');background-size:cover;}}
.earth::before{{top:3em;right:0;width:5em;height:5em;background:url('{earth_img}');background-size:cover;}}
.moon::before{{top:0.8em;right:0.2em;width:1.2em;height:1.2em;background:url('{moon_img}');background-size:cover;}}
.mars::before{{top:5em;right:3em;width:3em;height:3em;background:url('{mars_img}');background-size:cover;}}
.jupiter::before{{top:6em;right:3em;width:5em;height:5em;background:url('{jupiter_img}');background-size:cover;}}
.saturn::before{{top:7.5em;right:5em;width:4.5em;height:4.5em;background:url('{saturn_img}');background-size:cover;}}
.uranus::before{{top:9em;right:6.5em;width:4em;height:4em;background:url('{uranus_img}');background-size:cover;}}
.neptune::before{{top:10em;right:8em;width:4em;height:4em;background:url('{neptune_img}');background-size:cover;}}
.pluto::before{{top:11em;right:10em;width:4em;height:4em;background:url('{pluto_img}');background-size:cover;}}

.star{{position:absolute;background:white;border-radius:50%;z-index:-2;}}
@keyframes orbit {{to{{transform:rotate(360deg);}}}}
</style>

<script>
function createStars(){{
  const container=document.querySelector("body");
  for(let i=0;i<500;i++){{
    const star=document.createElement("div");
    star.className="star";
    star.style.width=".8px";star.style.height=".8px";
    star.style.top=Math.random()*100+"%";
    star.style.left=Math.random()*100+"%";
    container.appendChild(star);
  }}
}}
createStars();
</script>
"""

# ✅ Insert Background
components.html(solar_system_html, height=800)

# ---------- Page Content ----------
st.markdown("""
<div class="main-header">
    <h1>🌍🚀 AI Earth Lens</h1>
    <h3>A World Away: Hunting for Exoplanets with AI</h3>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.markdown('<h2 class="section-header">👨‍👩‍👦 Meet Our Team</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="team-card">', unsafe_allow_html=True)
    salem_img = load_image_as_base64("images/Salem.jpg")
    if salem_img:
        st.image(base64.b64decode(salem_img), width=200)
    st.markdown("**Salem Ahmed Salem**  \nTeam Leader | AI Developer | Web Developer")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="team-card">', unsafe_allow_html=True)
    ziad_img = load_image_as_base64("images/Ziad.png")
    if ziad_img:
        st.image(base64.b64decode(ziad_img), width=200)
    st.markdown("**Ziad Emad Ahmed**  \nResearcher | Web Developer")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.markdown('<h2 class="section-header">📌 About the Project</h2>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    Our project leverages Deep Learning to analyze astronomical data from <b>Kepler, K2, and TESS</b> telescopes.  
    <br><br>
    <p>1- Detect and classify exoplanets (Confirmed, Candidate, False Positive) </p>
    <p>2- Provide interactive orbit simulations 🌌  </p>
    <p>3- Enable both scientists and amateurs to explore discoveries with an intuitive UI 🚀 </p> 
</div>
""", unsafe_allow_html=True)