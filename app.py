import cv2
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import time
import plotly.graph_objects as go

# Import local logic
from symmetry_logic import get_landmark_coords_3d, get_asymmetry_scores, draw_asymmetry_overlays, FACEMESH_TESSELATION, LANDMARK_GROUPS

#APP CONFIGURATION
st.set_page_config(
    page_title="Facial Asymmetry Analysis | High-Precision Diagnostic System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional UI Styling
st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
    .main-header { font-size: 2.2rem; font-weight: 800; color: #1e293b; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1rem; color: #64748b; margin-bottom: 2rem; }
    .card { background-color: #ffffff; padding: 24px; border-radius: 16px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); border: 1px solid #f1f5f9; margin-bottom: 20px; }
    .metric-value { font-size: 1.8rem; font-weight: 800; color: #0f172a; }
    .badge { padding: 6px 12px; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; }
    .badge-perfect { background-color: #dcfce7; color: #166534; }
    .badge-mild { background-color: #dbeafe; color: #1e40af; }
    .badge-moderate { background-color: #fef3c7; color: #92400e; }
    .badge-severe { background-color: #fee2e2; color: #991b1b; }
    .step-header { font-size: 1.25rem; font-weight: 700; color: #334155; margin-bottom: 1rem; display: flex; align-items: center; gap: 10px; }
    .step-number { background-color: #3b82f6; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.875rem; }
    </style>
    """, unsafe_allow_html=True)

#SESSION STATE
if 'step' not in st.session_state: st.session_state.step = 1
if 'image' not in st.session_state: st.session_state.image = None
if 'coords_3d' not in st.session_state: st.session_state.coords_3d = None
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None

#SIDEBAR
with st.sidebar:
    st.markdown("<h2 style='color: #1e293b;'>Facial Asymmetry Analysis</h2>", unsafe_allow_html=True)
    st.caption("High-Precision Diagnostic System")
    st.markdown("---")
    
    st.subheader("System Controls")
    conf_threshold = st.slider("Detection Sensitivity", 0.1, 1.0, 0.5, 0.05)
    point_density = st.slider("Landmark Point Density", 0.1, 1.0, 1.0, 0.1)
    
    st.markdown("---")
    st.subheader("Visualization Mode")
    viz_mode = st.radio("Photo Overlay Style", ["Landmark Points Only", "Points + Feature Detection"], index=0)
    show_features = (viz_mode == "Points + Feature Detection")
    
    st.markdown("---")
    if st.button("Reset Analysis", use_container_width=True):
        st.session_state.step = 1
        st.session_state.image = None
        st.session_state.coords_3d = None
        st.session_state.analysis_results = None
        st.rerun()

#MAIN CONTENT
MODEL_PATH = 'face_landmarker.task'

st.markdown("<div class='main-header'>Facial Asymmetry Analysis</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>High-precision diagnostic workflow with advanced face features detection.</div>", unsafe_allow_html=True)

#UPLOAD
if st.session_state.step == 1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='step-header'><div class='step-number'>1</div> Patient Image Input</div>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload high-resolution facial image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        st.session_state.image = np.array(Image.open(uploaded_file))
        st.image(st.session_state.image, caption="Patient Image", width=400)
        
        h, w, _ = st.session_state.image.shape
        st.markdown(f"**Specifications:** {w}x{h} pixels | Format: {uploaded_file.type}")
        
        if st.button("Generate Diagnostic Overlay", type="primary"):
            st.session_state.step = 2
            st.rerun()
    else:
        st.info("Please upload a patient image to begin.")
    st.markdown("</div>", unsafe_allow_html=True)

#POINT OVERLAY
elif st.session_state.step == 2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='step-header'><div class='step-number'>2</div> High-Precision Diagnostic Overlay</div>", unsafe_allow_html=True)
    
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1, min_face_detection_confidence=conf_threshold)
    
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=st.session_state.image)
        result = landmarker.detect(mp_image)
        
        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            h, w, _ = st.session_state.image.shape
            
            class MockLM:
                def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z
            class MockList:
                def __init__(self, lms): self.landmark = [MockLM(l.x, l.y, l.z) for l in lms]
            
            st.session_state.coords_3d = get_landmark_coords_3d(MockList(landmarks), w, h)
            overlay_img, _ = draw_asymmetry_overlays(st.session_state.image, st.session_state.coords_3d, {"None": 0}, show_points=True, point_density=point_density, show_features=show_features)
            st.image(overlay_img, caption="Diagnostic Overlay on Patient Photo", use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("View Interactive 3D Mesh", use_container_width=True):
                    st.session_state.step = 3
                    st.rerun()
            with col2:
                if st.button("Calculate Asymmetry", type="primary", use_container_width=True):
                    st.session_state.step = 4
                    st.rerun()
        else:
            st.error("Face detection failed.")
            if st.button("Back to Upload"):
                st.session_state.step = 1
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

#INTERACTIVE 3D
elif st.session_state.step == 3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='step-header'><div class='step-number'>3</div> Interactive 3D Reconstruction</div>", unsafe_allow_html=True)
    
    coords = st.session_state.coords_3d
    h, w, _ = st.session_state.image.shape
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    y = h - y 
    
    fig = go.Figure()
    num_connections = len(FACEMESH_TESSELATION)
    edge_x, edge_y, edge_z = [], [], []
    for i in range(num_connections):
        p1, p2 = FACEMESH_TESSELATION[i]
        edge_x.extend([x[p1], x[p2], None])
        edge_y.extend([y[p1], y[p2], None])
        edge_z.extend([z[p1], z[p2], None])
    
    fig.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(color='silver', width=1), hoverinfo='none'))
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=2, color='blue', opacity=0.8), hoverinfo='none'))
    fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0), height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    if st.button("Proceed to Analysis", type="primary", use_container_width=True):
        st.session_state.step = 4
        st.rerun()
    if st.button("Back to Overlay", use_container_width=True):
        st.session_state.step = 2
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

#ANALYZE
elif st.session_state.step == 4:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='step-header'><div class='step-number'>4</div> High-Precision Analysis</div>", unsafe_allow_html=True)
    
    coords = st.session_state.coords_3d
    scores, total, ai = get_asymmetry_scores(coords)
    out_img, worst = draw_asymmetry_overlays(st.session_state.image, coords, scores, show_points=True, point_density=point_density, show_features=show_features)
    
    st.session_state.analysis_results = {"scores": scores, "total": total, "ai": ai, "worst": worst, "image": out_img}
    
    col_viz, col_metrics = st.columns([3, 2])
    
    with col_viz:
        st.image(out_img, caption="Diagnostic Analysis Overlay", use_container_width=True)
        categories = list(scores.keys())
        values = list(scores.values())
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]], fill='toself', line_color='#3b82f6', fillcolor='rgba(59, 130, 246, 0.2)'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1000])), showlegend=False, height=400, margin=dict(l=40, r=40, t=40, b=40))
        st.plotly_chart(fig, use_container_width=True)

    with col_metrics:
        st.markdown("### Clinical Metrics (0-1000)")
        m1, m2 = st.columns(2)
        m1.markdown(f"<div class='metric-label'>Overall Score</div><div class='metric-value'>{total:.1f}</div>", unsafe_allow_html=True)
        
        if total < 100: sev, badge = "Perfect", "badge-perfect"
        elif total < 350: sev, badge = "Mild", "badge-mild"
        elif total < 650: sev, badge = "Moderate", "badge-moderate"
        else: sev, badge = "Severe", "badge-severe"
        
        m2.markdown(f"<div class='metric-label'>Severity Status</div><span class='badge {badge}'>{sev}</span>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("#### Regional Breakdown")
        for region, score in scores.items():
            st.write(f"**{region}**")
            st.progress(score / 1000.0)
            st.caption(f"Precision Score: {score:.1f}/1000.0")
            
        st.markdown("---")
        st.info(f"**Primary Finding:** The **{worst}** region exhibits the highest structural deviation.")
        
        if st.button("Generate Final Report", type="primary", use_container_width=True):
            st.session_state.step = 5
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

#REPORT
elif st.session_state.step == 5:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='step-header'><div class='step-number'>5</div> Clinical Diagnostic Report</div>", unsafe_allow_html=True)
    
    res = st.session_state.analysis_results
    st.success("High-precision analysis complete. Report ready for export.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(res["image"], caption="Final Diagnostic Snapshot", use_container_width=True)
    
    with col2:
        st.markdown("### Patient Diagnostic Summary")
        st.write(f"- **Overall Asymmetry Score:** {res['total']:.1f}/1000")
        st.write(f"- **Asymmetry Index (AI):** {res['ai']:.2f}%")
        st.write(f"- **Primary Site of Deviation:** {res['worst']}")
        
        st.markdown("---")
        st.markdown("### Export Options")
        df = pd.DataFrame(list(res["scores"].items()), columns=['Region', 'Score (0-1000)'])
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download High-Precision CSV Report", data=csv, file_name='facial_analysis_report.csv', mime='text/csv', use_container_width=True)
        
        if st.button("New Patient Analysis", use_container_width=True):
            st.session_state.step = 1
            st.session_state.image = None
            st.session_state.coords_3d = None
            st.session_state.analysis_results = None
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Facial Asymmetry Prototype")
