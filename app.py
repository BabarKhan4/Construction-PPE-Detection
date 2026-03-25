import cv2
import numpy as np
import os
import time
import base64
import streamlit as st
from PIL import Image

from services.yolo_service import YoloService
from services.video_processor import VideoProcessor
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(page_title="GEAR - Guardian Eye for At-Risk workers", layout="wide")

# ----------------- CUSTOM CSS ANIMATIONS & BOOTSTRAP -----------------
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Global Animations */
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 5px #dc2626; }
        50% { box-shadow: 0 0 20px #dc2626; }
        100% { box-shadow: 0 0 5px #dc2626; }
    }

    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        text-align: center;
    }
    .main-header h1 {
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: white;
    }
    
    /* Fixed Media Container */
    .media-box {
        background-color: #0f172a;
        border-radius: 12px;
        padding: 10px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        display: flex;
        justify-content: center;
        align-items: center;
    }

    /* Metric Cards */
    .metric-card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.2s;
        margin-bottom: 1rem;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px rgba(0,0,0,0.1);
    }
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
        color: #2a5298;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1a1a1a;
    }

    /* Log Bubbles */
    .log-container {
        height: 480px;
        overflow-y: auto;
        padding: 10px;
        background: #f8fafc;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    .log-bubble {
        background: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border-left: 4px solid #3b82f6;
        animation: slideInRight 0.4s ease-out forwards;
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .log-bubble.violation {
        border-left: 4px solid #ef4444;
        background: #fef2f2;
        animation: slideInRight 0.4s ease-out forwards, pulseGlow 2s infinite;
    }
    .log-img-crop {
        width: 60px;
        height: 60px;
        object-fit: cover;
        border-radius: 6px;
        border: 1px solid #cbd5e1;
    }
    .log-text {
        font-size: 0.9rem;
        color: #334155;
        font-weight: 600;
    }
    .log-timestamp {
        font-size: 0.75rem;
        color: #94a3b8;
        display: block;
        margin-top: 4px;
    }
    
    /* Streamlit overrides */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------- LOAD MODEL -----------------
@st.cache_resource
def load_model():
    return YoloService("Model/ppe.pt")

yolo_service = load_model()

# ----------------- HELPERS -----------------
def render_metric(icon, value, label):
    return f"""
    <div class="metric-card">
        <div class="metric-icon"><i class="fa-solid {icon}"></i></div>
        <div class="metric-value">{value}</div>
        <div style="font-size: 0.9rem; color: #666; text-transform: uppercase;">{label}</div>
    </div>
    """

def numpy_to_base64(img_array):
    try:
        _, buffer = cv2.imencode('.jpg', img_array)
        encoded_string = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded_string}"
    except:
        return ""

def generate_log_html(logs, crops):
    html = '<div class="log-container">'
    
    # Render Violations First (Cropped Array Visuals)
    for crop in reversed(crops[-5:]): # Show last 5 crops
        b64_img = numpy_to_base64(crop)
        ts = time.strftime("%H:%M:%S")
        html += '<div class="log-bubble violation">'
        html += f'<img src="{b64_img}" class="log-img-crop">'
        html += '<div><span class="log-text"><i class="fa-solid fa-triangle-exclamation"></i> Privacy Violation</span>'
        html += f'<span class="log-timestamp">{ts} | Missing Hardhat</span></div></div>'
        
    # Render Standard String Logs
    for log in reversed(logs[-10:]):
        ts = time.strftime("%H:%M:%S")
        html += '<div class="log-bubble">'
        html += f'<div><span class="log-text"><i class="fa-solid fa-satellite-dish"></i> {log}</span>'
        html += f'<span class="log-timestamp">{ts} | AI Vision Core</span></div></div>'
        
    html += '</div>'
    return html

# ----------------- UI HEADER -----------------
st.markdown("""
<div class="main-header">
    <h1><i class="fa-solid fa-helmet-safety"></i> GEAR - Guardian Eye for At-Risk workers</h1>
    <p>Real-time machine learning oversight for Personal Protective Equipment compliance.</p>
</div>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.markdown('<h3><i class="fa-solid fa-sliders"></i> Configuration</h3>', unsafe_allow_html=True)
mode = st.sidebar.selectbox("Select Detection Mode", ["Static Image", "Live Webcam", "YouTube Video"], index=0)

st.sidebar.markdown('<hr style="margin: 10px 0;">', unsafe_allow_html=True)
st.sidebar.markdown('#### <i class="fa-solid fa-wand-magic-sparkles"></i> Core Overrides', unsafe_allow_html=True)

conf_threshold = st.sidebar.slider("AI Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
box_style = st.sidebar.selectbox("Tracking Stylizer", ["Standard", "Minimalist", "Cyberpunk"])
privacy_mode = st.sidebar.toggle("Enable Privacy Blurring", value=True, help="Automatically blurs the faces/bodies of personnel caught missing compliance gear.")

st.sidebar.markdown('#### <i class="fa-solid fa-filter"></i> Tensor Tracking Classes', unsafe_allow_html=True)
c_hh = st.sidebar.checkbox("Hardhats", value=True)
c_sv = st.sidebar.checkbox("Safety Vests", value=True)
c_p = st.sidebar.checkbox("Personnel", value=True)

active_classes = []
if c_hh: active_classes.extend(["Hardhat", "NO-Hardhat"])
if c_sv: active_classes.extend(["Safety Vest", "NO-Safety Vest"])
if c_p: active_classes.append("Person")
if len(active_classes) == 0: active_classes = ["Hardhat"] # fallback

# ----------------- CORE LOGIC -----------------
if mode == "Static Image":
    st.markdown('<h3><i class="fa-regular fa-image"></i> Static Image Inspection</h3>', unsafe_allow_html=True)
    option = st.radio("Input Source", ("Select Example", "Upload Own"), horizontal=True)
    
    image_source = None
    if option == "Select Example":
        example_folder = "test_images"
        if os.path.exists(example_folder):
            examples = sorted([f for f in os.listdir(example_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if examples:
                selected_example = st.selectbox("Choose an example image", examples)
                image_source = os.path.join(example_folder, selected_example)
    elif option == "Upload Own":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image_source = uploaded_file

    if image_source is not None:
        try:
            pil_image = Image.open(image_source)
        except Exception:
            st.error("Failed to load image.")
            st.stop()
            
        img_array = np.array(pil_image)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR if img_array.shape[-1] == 3 else cv2.COLOR_RGBA2BGR)
        elif len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            
        # Force uniform mathematical aspect ratio
        img_array = cv2.resize(img_array, (854, 480))

        with st.spinner("Executing tensor calculations..."):
            annotated_img, stats, hh_det, p_det, evt_logs, violation_crops = yolo_service.predict_and_annotate(
                frame=img_array.copy(),
                conf_threshold=conf_threshold,
                selected_classes=active_classes,
                box_style=box_style,
                privacy_mode=privacy_mode
            )
            
        metrics_col, frame_col, log_col = st.columns([1, 2.5, 1.2])
        
        with metrics_col:
            st.markdown(render_metric("fa-hard-hat", stats.get("Hardhat", 0), "Hardhats"), unsafe_allow_html=True)
            st.markdown(render_metric("fa-vest", stats.get("Safety Vest", 0), "Safety Vests"), unsafe_allow_html=True)
            st.markdown(render_metric("fa-users", stats.get("Person", 0), "Personnel"), unsafe_allow_html=True)
            
        with frame_col:
            annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            st.markdown('<div class="media-box">', unsafe_allow_html=True)
            st.image(annotated_rgb, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with log_col:
            st.markdown('#### <i class="fa-solid fa-bars-staggered"></i> Live Incident Log', unsafe_allow_html=True)
            st.markdown(generate_log_html(evt_logs, violation_crops), unsafe_allow_html=True)
            
elif mode == "Live Webcam":
    st.markdown('<h3><i class="fa-solid fa-video"></i> Live WebRTC Feed</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<div class="media-box">', unsafe_allow_html=True)
        ctx = webrtc_streamer(
            key="ppe-detection-stream",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if ctx.video_processor:
            # Inject UI State dynamically into the WebRTC Thread
            ctx.video_processor.conf_threshold = conf_threshold
            ctx.video_processor.selected_classes = active_classes
            ctx.video_processor.box_style = box_style
            ctx.video_processor.privacy_mode = privacy_mode
            
    with col2:
        st.info("Interactive widgets (Confidence, Blur, Stylization) actively push configuration shifts into the WebRTC VideoProcessor thread in real-time.")
    
elif mode == "YouTube Video":
    st.markdown('<h3><i class="fa-brands fa-youtube"></i> YouTube Active Stream</h3>', unsafe_allow_html=True)
    
    url = st.text_input("Stream Source (YouTube URL):")
    start_btn = st.button("Start Extrapolation")
    
    if url and start_btn:
        try:
            from vidgear.gears import CamGear
        except ImportError:
            st.error("Missing YouTube dependencies (`vidgear`, `yt-dlp`).")
            st.stop()
            
        st.info('💡 Adjust sidebar filters live! Click "Stop" in the top right corner natively to halt execution.')
        
        metrics_col, frame_col, log_col = st.columns([1, 2.5, 1.2])
        
        with metrics_col:
            hh_metric = st.empty()
            vest_metric = st.empty()
            p_metric = st.empty()
            
        with frame_col:
            st.markdown('<div class="media-box">', unsafe_allow_html=True)
            stframe = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)
            
        with log_col:
            st.markdown('#### <i class="fa-solid fa-bolt"></i> Live Target Traces', unsafe_allow_html=True)
            log_placeholder = st.empty()
        
        try:
            options = {"STREAM_RESOLUTION": "480p", "STREAM_PARAMS": {"nocheckcertificate": True}} 
            stream = CamGear(source=url, stream_mode=True, logging=True, **options).start()
            
            while True:
                frame = stream.read()
                if frame is None:
                    break
                    
                # Fix rendering box
                frame = cv2.resize(frame, (854, 480))
                    
                annotated_img, stats, hh_det, p_det, evt_logs, v_crops = yolo_service.predict_and_annotate(
                    frame=frame,
                    conf_threshold=conf_threshold,
                    selected_classes=active_classes,
                    box_style=box_style,
                    privacy_mode=privacy_mode
                )
                
                # Render Metrics
                hh_metric.markdown(render_metric("fa-hard-hat", stats.get("Hardhat", 0), "Hardhats"), unsafe_allow_html=True)
                vest_metric.markdown(render_metric("fa-vest", stats.get("Safety Vest", 0), "Safety Vests"), unsafe_allow_html=True)
                p_metric.markdown(render_metric("fa-users", stats.get("Person", 0), "Personnel"), unsafe_allow_html=True)

                # Render Fixed Container Image
                annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                stframe.image(annotated_rgb, use_container_width=True)
                
                # Render Animated Log Bubbles
                log_placeholder.markdown(generate_log_html(evt_logs, v_crops), unsafe_allow_html=True)
                
            stream.stop()
        except Exception as e:
            st.error(f"Stream error: {e}")
