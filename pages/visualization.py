import streamlit as st
from pathlib import Path
import base64

st.set_page_config(
    page_title="Kepler Transit Simulator",
    page_icon="🌌",
    layout="wide"
)

st.title("🌌 Kepler Transit Simulator")

st.markdown(
    """
    Click 'Start Simulation' to see the planet transit the star. 
    Use 'Stop Simulation' to pause it completely.
    """
)

# Initialize session state for simulation status
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False

# Simulation controls
col1, col2 = st.columns(2)
with col1:
    start = st.button("Start Simulation")
with col2:
    stop = st.button("Stop Simulation")

# Update session state based on button clicks
if start:
    st.session_state.simulation_running = True
if stop:
    st.session_state.simulation_running = False

# File paths
star_path = Path("star.glb")
planet_path = Path("planet.glb")
background_path = Path("background.jpg")
sound_path = Path("comment.mp3")

if not star_path.exists() or not planet_path.exists() or not sound_path.exists():
    st.error("star.glb, planet.glb, and comment.mp3 must exist in the project folder.")
elif st.session_state.simulation_running or start or stop:
    def file_to_base64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    star_b64 = file_to_base64(star_path)
    planet_b64 = file_to_base64(planet_path)
    sound_b64 = file_to_base64(sound_path)

    orbit_width = 600
    orbit_height = 200
    star_width = 150
    planet_width = 40

    background_style = (
        f"background-image: url('data:image/jpeg;base64,{file_to_base64(background_path)}');"
        "background-size: cover; background-position: center;" 
        if background_path.exists() else 
        "background: radial-gradient(circle, #0a0e2a 0%, #000 70%);"
    )

    # Pass simulation state to JS
    simulation_running = "true" if st.session_state.simulation_running else "false"

    html_code = f"""
    <style>
        .container {{
            width: {orbit_width}px;
            height: {orbit_height}px;
            margin: auto;
            position: relative;
            {background_style}
        }}
        #star {{
            position: absolute;
            top: 50%;
            left: 50%;
            width: {star_width}px;
            height: {star_width}px;
            transform: translate(-50%, -50%);
            z-index: 2;
        }}
        #planet {{
            position: absolute;
            width: {planet_width}px;
            height: {planet_width}px;
            transform: translateY(-50%);
            z-index: 3;
            transition: opacity 0.1s linear;
        }}
        .status {{
            text-align: center;
            margin-top: 10px;
            font-weight: bold;
            color: #fff;
        }}
    </style>

    <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>

    <div class="container">
        <model-viewer id="star" src="data:model/gltf-binary;base64,{star_b64}" 
            camera-controls auto-rotate="false">
        </model-viewer>

        <model-viewer id="planet" src="data:model/gltf-binary;base64,{planet_b64}" 
            camera-controls auto-rotate="false">
        </model-viewer>

        <audio id="commentAudio">
            <source src="data:audio/mp3;base64,{sound_b64}" type="audio/mp3">
        </audio>
    </div>
    <div class="status" id="status">Status: { "Running" if st.session_state.simulation_running else "Stopped" }</div>

    <script>
        const planet = document.getElementById('planet');
        const audio = document.getElementById('commentAudio');
        const statusDiv = document.getElementById('status');
        const containerWidth = {orbit_width};
        const containerHeight = {orbit_height};
        const starWidth = {star_width};
        const planetWidth = {planet_width};
        const centerY = containerHeight / 2;
        const starCenterX = containerWidth / 2;

        let x = -planetWidth;
        let y = centerY;
        const speedX = 1.5;
        const driftY = 2;
        let paused = false;
        let hasPausedThisPass = false;
        let animationId = null;

        // Simulation state from Streamlit
        let simulationRunning = {simulation_running};

        function resetSimulation() {{
            x = -planetWidth;
            y = centerY;
            paused = false;
            hasPausedThisPass = false;
            planet.style.left = x + 'px';
            planet.style.top = y + 'px';
            planet.style.opacity = 0.0;
            
            // Stop audio if playing
            if (!audio.paused) {{
                audio.pause();
                audio.currentTime = 0;
            }}
        }}

        function stopSimulation() {{
            if (animationId) {{
                cancelAnimationFrame(animationId);
                animationId = null;
            }}
            resetSimulation();
            statusDiv.textContent = "Status: Stopped";
        }}

        function animateTransit() {{
            if (simulationRunning) {{
                if (!paused) {{
                    x += speedX;
                    y = centerY + Math.sin(x / 50) * driftY;

                    planet.style.left = x + 'px';
                    planet.style.top = y + 'px';

                    const planetCenter = x + planetWidth / 2;

                    // Opacity
                    if (planetCenter > starCenterX - starWidth / 2 && planetCenter < starCenterX + starWidth / 2) {{
                        planet.style.opacity = 1.0;
                    }} else {{
                        planet.style.opacity = 0.0;
                    }}

                    // Pause at center once
                    if (!hasPausedThisPass && planetCenter >= starCenterX) {{
                        paused = true;
                        x = starCenterX - planetWidth / 2;
                        planet.style.left = x + 'px';
                        audio.play();
                        hasPausedThisPass = true;
                        audio.onended = () => {{
                            paused = false;
                        }};
                    }}

                    // Loop
                    if (x > containerWidth) {{
                        x = -planetWidth;
                        hasPausedThisPass = false;
                    }}
                }}
                animationId = requestAnimationFrame(animateTransit);
            }} else {{
                stopSimulation();
            }}
        }}

        // Initialize simulation state
        if (simulationRunning) {{
            statusDiv.textContent = "Status: Running";
            animateTransit();
        }} else {{
            stopSimulation();
        }}

        // Listen for changes in simulation state (when Streamlit re-runs)
        window.addEventListener('load', function() {{
            // Reset and start/stop based on current state
            if (!simulationRunning) {{
                stopSimulation();
            }}
        }});
    </script>
    """

    st.components.v1.html(html_code, height=orbit_height + 80)
else:
    st.info("Click 'Start Simulation' to begin the planet transit animation.")