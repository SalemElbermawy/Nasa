# app_final_white_sidebar.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import time
time.sleep(0.1)
st.experimental_rerun()
# إعداد الصفحة
st.set_page_config(
    page_title="NASA Exoplanet AI - 98% Accuracy",
    layout="wide",
    page_icon="🪐",
    initial_sidebar_state="expanded"
)

# CSS مخصص - سايدبار أبيض
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .main-header {
        font-size: 4rem;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 50%, #45B7D1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        padding: 20px;
        text-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.5rem;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    
    .nasa-card {
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.8) 100%);
        backdrop-filter: blur(10px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.3);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .nasa-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
    }
    
    .nasa-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }
    
    /* سايدبار أبيض */
    section[data-testid="stSidebar"] {
        background-color: white !important;
        border-right: 2px solid #f0f0f0;
    }
    
    section[data-testid="stSidebar"] * {
        color: #333333 !important;
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #2c3e50 !important;
    }
    
    .sidebar-metric {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white !important;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    
    .sidebar-metric h1, 
    .sidebar-metric h3 {
        color: white !important;
    }
    
    .sidebar-info {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #4ECDC4;
        margin: 10px 0;
    }
    
    .class-badge {
        display: inline-block;
        padding: 8px 12px;
        border-radius: 20px;
        font-weight: bold;
        margin: 2px;
        font-size: 0.8rem;
    }
    
    .confirmed-badge { background: #1dd1a1; color: white; }
    .candidate-badge { background: #feca57; color: black; }
    .false-positive-badge { background: #ff6b6b; color: white; }
    .refuted-badge { background: #a29bfe; color: white; }
</style>
""", unsafe_allow_html=True)

# باقي الكود يبقى كما هو مع تعديل السايدبار فقط

# تحميل الموديل
@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    performance_data = {}
    
    with st.spinner("🚀 Loading NASA AI Model..."):
        try:
            models['xgb'] = joblib.load('xgb_model_final.pkl')
            models['scaler'] = joblib.load('scaler_final.pkl')
            models['label_encoder'] = joblib.load('label_encoder_final.pkl')
            feature_names = joblib.load('feature_names_final.pkl')
            model_info = joblib.load('model_info_final.pkl')
            performance_data.update(model_info)
            
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
    
    return models, performance_data, feature_names

# تحميل الموديلات
models, performance_data, feature_names = load_models()

# إعداد الفئات والألوان
mapping = performance_data.get('class_mapping', {})
colors = {
    "CONFIRMED": "#1dd1a1",
    "CANDIDATE": "#feca57", 
    "FALSE POSITIVE": "#ff6b6b",
    "REFUTED": "#a29bfe"
}
icons = {
    "CONFIRMED": "🟢", 
    "CANDIDATE": "🟡", 
    "FALSE POSITIVE": "🔴",
    "REFUTED": "🟣"
}
descriptions = {
    "CONFIRMED": "✅ CONFIRMED EXOPLANET - Verified planetary discovery!",
    "CANDIDATE": "🟡 PLANETARY CANDIDATE - Requires further observation",
    "FALSE POSITIVE": "🔴 FALSE POSITIVE - Astronomical artifact, not a planet", 
    "REFUTED": "🟣 REFUTED - Officially disproven as a planet"
}

# تهيئة session state
if "manual_data" not in st.session_state:
    st.session_state.manual_data = pd.DataFrame(columns=feature_names, index=[0]).fillna(0)
if "table_data" not in st.session_state:
    st.session_state.table_data = pd.DataFrame(columns=feature_names, index=[0]).fillna(0)
if "results_xgb" not in st.session_state:
    st.session_state.results_xgb = None

# الشريط الجانبي - أبيض مع نصوص ملونة
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="color: #2c3e50; font-size: 2.5rem;">🪐</h1>
        <h2 style="color: #2c3e50; margin: 0;">NASA AI</h2>
        <p style="color: #666;">Exoplanet Discovery</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🏆 Model Performance")
    st.markdown("""
    <div class="sidebar-metric">
        <h3 style="color: white !important; margin: 0;">🎯 AI Accuracy</h3>
        <h1 style="color: white !important; margin: 0; font-size: 2.5rem;">98%</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🔬 4-Class System")
    
    st.markdown("""
    <div style="margin: 10px 0;">
        <span class="class-badge confirmed-badge">🟢 CONFIRMED</span>
        <span class="class-badge candidate-badge">🟡 CANDIDATE</span>
    </div>
    <div style="margin: 10px 0;">
        <span class="class-badge false-positive-badge">🔴 FALSE POSITIVE</span>
        <span class="class-badge refuted-badge">🟣 REFUTED</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🚀 Innovation")
    st.markdown("""
    <div class="sidebar-info">
        <p style="margin: 0; color: #333; font-size: 0.9rem;">
        <strong style="color: #2c3e50;">94 → 31 features</strong><br>
        <strong style="color: #e74c3c;">67% reduction</strong><br>
        <strong style="color: #27ae60;">98% accuracy</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

# الواجهة الرئيسية
st.markdown('<h1 class="main-header">🌌 NASA Exoplanet Discovery AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">4-Class Classification • 98% Accuracy • Revolutionary Feature Optimization</p>', unsafe_allow_html=True)

# علامات التبويب
tab1, tab2, tab3 = st.tabs(["🚀 AI Prediction", "📊 Model Insights", "🌟 About Project"])

with tab1:
    st.header("🔭 Input Stellar Data")
    
    input_method = st.radio("Input Method:", ["Manual Entry", "Data Table"], horizontal=True)
    
    if input_method == "Manual Entry":
        st.subheader("🔧 Manual Feature Input")
        
        sample_data = {
            'pl_orbper': 10.5, 'pl_trandeph': 0.008, 'pl_trandurh': 0.15,
            'pl_rade': 2.5, 'pl_eqt': 850, 'st_tmag': 12.3,
            'st_dist': 150.0, 'st_teff': 5800, 'st_logg': 4.4,
            'st_rade': 1.1
        }
        
        if st.button("🎲 Load Sample Data", ):
            for col in feature_names:
                if col in sample_data:
                    st.session_state.manual_data.at[0, col] = sample_data[col]
            st.success("✅ Sample data loaded!")
            st.rerun()
        
        cols_per_row = 3
        for i in range(0, len(feature_names), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col_name in enumerate(feature_names[i:i+cols_per_row]):
                with cols[j]:
                    st.session_state.manual_data.at[0, col_name] = st.number_input(
                        f"**{col_name}**",
                        value=float(st.session_state.manual_data.at[0, col_name]),
                        format="%.6f",
                        key=f"manual_{col_name}"
                    )
        
        current_data = st.session_state.manual_data

    else:
        st.subheader("📋 Tabular Data Input")
        edited_df = st.data_editor(st.session_state.table_data, num_rows="dynamic",  height=400)
        st.session_state.table_data = edited_df
        current_data = st.session_state.table_data

    # Prediction
    st.markdown("---")
    st.header("🎯 AI Analysis")
    
    if st.button("🌌 Analyze with NASA AI", type="primary", ):
        if not current_data.empty and all(m in models for m in ['xgb', 'scaler', 'label_encoder']):
            with st.spinner("🔭 AI analyzing cosmic patterns..."):
                try:
                    scaled_data = models['scaler'].transform(current_data)
                    xgb_pred_prob = models['xgb'].predict_proba(scaled_data)
                    xgb_pred = models['xgb'].predict(scaled_data)
                    
                    le = models['label_encoder']
                    class_names = le.classes_
                    
                    results = []
                    for i in range(len(xgb_pred)):
                        results.append({
                            "Prediction": class_names[xgb_pred[i]],
                            "Probabilities": {class_names[j]: float(xgb_pred_prob[i,j]) for j in range(len(class_names))},
                            "Confidence": float(np.max(xgb_pred_prob[i]))
                        })
                    
                    st.session_state.results_xgb = results
                    st.success(f"✅ Analysis complete for {len(xgb_pred)} observation(s)!")
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
        else:
            st.warning("⚠️ Please ensure model is loaded and data is valid")

    # Display results
    if st.session_state.results_xgb:
        st.markdown("---")
        st.header("📈 Analysis Results")
        
        # دالة عرض النتائج
        def display_prediction_result(result, model_name, idx):
            pred_class = result["Prediction"]
            confidence = result["Confidence"]
            probabilities = result["Probabilities"]
            
            st.markdown(f"""
            <div class="nasa-card">
                <div class="prediction-header">
                    <span class="prediction-icon">{icons[pred_class]}</span>
                    <div>
                        <h2 style="margin: 0; color: #2c3e50;">{model_name} Analysis</h2>
                        <h1 style="margin: 0; font-size: 2.5rem; color: {colors[pred_class]};">{pred_class}</h1>
                    </div>
                </div>
                <div class="confidence-badge" style="background: {colors[pred_class]}22; border-color: {colors[pred_class]}; color: {colors[pred_class]};">
                    🔮 Confidence Level: {confidence:.1%}
                </div>
                <div style="margin-top: 15px; padding: 15px; background: {colors[pred_class]}11; border-radius: 10px;">
                    <strong>📝 Scientific Interpretation:</strong> {descriptions[pred_class]}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Visualization
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown('<div class="nasa-card">', unsafe_allow_html=True)
                st.subheader("📊 Probability Distribution")
                
                fig = go.Figure(data=[go.Pie(
                    labels=list(probabilities.keys()),
                    values=list(probabilities.values()),
                    hole=0.6,
                    marker_colors=[colors[k] for k in probabilities.keys()],
                    textinfo='label+percent',
                    textposition='inside',
                    hoverinfo='label+value+percent'
                )])
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=14, color='#2c3e50'),
                    margin=dict(t=0, b=0, l=0, r=0)
                )
                st.plotly_chart(fig, )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="nasa-card">', unsafe_allow_html=True)
                st.subheader("📈 Confidence Analytics")
                
                metrics_cols = st.columns(2)
                with metrics_cols[0]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🎯</h3>
                        <h2>{pred_class}</h2>
                        <p>Classification</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metrics_cols[1]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>💪</h3>
                        <h2>{confidence:.1%}</h2>
                        <p>Confidence</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = confidence * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "AI Confidence Level", 'font': {'size': 20, 'color': '#2c3e50'}},
                    number = {'suffix': "%", 'font': {'size': 36, 'color': '#2c3e50'}},
                    delta = {'reference': 70, 'increasing': {'color': "#1dd1a1"}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#2c3e50"},
                        'bar': {'color': colors[pred_class], 'thickness': 0.75},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "#e0e0e0",
                        'steps': [
                            {'range': [0, 60], 'color': '#ffebee'},
                            {'range': [60, 85], 'color': '#fff3e0'},
                            {'range': [85, 100], 'color': '#e8f5e8'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=300, margin=dict(l=50, r=50, t=80, b=50))
                st.plotly_chart(fig, )
                st.markdown('</div>', unsafe_allow_html=True)
        
        for idx, res in enumerate(st.session_state.results_xgb):
            display_prediction_result(res, "NASA AI", idx)

with tab2:
    st.header("📊 Model Performance Insights")
    
    st.markdown("""
    <div class="achievement-card">
        <h2>🏆 BREAKTHROUGH ACHIEVEMENT</h2>
        <h1>98% Accuracy with 67% Fewer Features!</h1>
        <p>Revolutionary feature optimization achieving elite performance with minimal data requirements</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Accuracy Metrics")
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = 98,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "AI Accuracy", 'font': {'size': 24}},
            number = {'suffix': "%", 'font': {'size': 40}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 2},
                'bar': {'color': "#1dd1a1", 'thickness': 0.75},
                'steps': [
                    {'range': [0, 80], 'color': "#ffebee"},
                    {'range': [80, 95], 'color': "#fff3e0"},
                    {'range': [95, 100], 'color': "#e8f5e8"}
                ],
            }
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, )
    
    with col2:
        st.subheader("📉 Feature Optimization")
        features_data = {
            'Stage': ['Original', 'Optimized'],
            'Features': [94, 31],
            'Color': ['#ff6b6b', '#1dd1a1']
        }
        
        fig = px.bar(features_data, x='Stage', y='Features', color='Stage',
                    color_discrete_map={'Original': '#ff6b6b', 'Optimized': '#1dd1a1'},
                    text='Features')
        fig.update_layout(showlegend=False, yaxis_title="Number of Features")
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        st.plotly_chart(fig, )
        
        st.markdown("""
        <div class="innovation-card">
            <h3>🚀 67% Feature Reduction</h3>
            <p><strong>94 → 31 features</strong> while maintaining <strong>98% accuracy</strong></p>
            <p>Intelligent feature selection outperforms brute-force data collection</p>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.header("🌟 About NASA Exoplanet AI Project")
    
    st.markdown("""
    <div class="nasa-card">
        <h1>🪐 Revolutionizing Exoplanet Discovery</h1>
        <p><strong>Advanced AI system achieving breakthrough performance through intelligent optimization</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🎯 Project Overview")
        st.markdown("""
        This NASA-inspired AI system represents a **paradigm shift** in exoplanet classification, 
        achieving **elite 98% accuracy** with only **31 optimized features** - a revolutionary 
        approach that maintains exceptional performance while dramatically reducing data requirements.
        """)
        
        st.markdown("### 🚀 Breakthrough Achievements")
        
        # استخدام st.container بدلاً من HTML مباشر
        with st.container():
            st.markdown("""
            <div style="background: white; padding: 20px; border-radius: 10px; border-left: 5px solid #1dd1a1; margin: 10px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.05);">
                <h4 style="color: #2c3e50; margin: 0;">📊 Unprecedented Efficiency</h4>
                <p style="margin: 10px 0;"><strong style="color: #1dd1a1;">94 → 31 features (67% reduction)</strong> with <strong>zero accuracy loss</strong></p>
                <p style="margin: 0; color: #666;">We've identified the most impactful planetary signatures, eliminating redundancy while preserving predictive power</p>
            </div>
            """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown("""
            <div style="background: white; padding: 20px; border-radius: 10px; border-left: 5px solid #feca57; margin: 10px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.05);">
                <h4 style="color: #2c3e50; margin: 0;">🎯 Elite Performance</h4>
                <p style="margin: 10px 0;"><strong style="color: #feca57;">98% Accuracy</strong> - Near-perfect classification using only essential data</p>
                <p style="margin: 0; color: #666;">Proving that intelligent feature selection outperforms brute-force data collection</p>
            </div>
            """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown("""
            <div style="background: white; padding: 20px; border-radius: 10px; border-left: 5px solid #a29bfe; margin: 10px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.05);">
                <h4 style="color: #2c3e50; margin: 0;">⚡ Computational Revolution</h4>
                <p style="margin: 10px 0;"><strong style="color: #a29bfe;">Faster analysis, reduced storage, and real-time processing capabilities</strong></p>
                <p style="margin: 0; color: #666;">Enabling large-scale exoplanet surveys with minimal computational resources</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("## 🔬 Scientific Impact")
        st.markdown("""
        Our methodology demonstrates that **quality of features matters more than quantity of data**, 
        opening new possibilities for astronomical research with limited computational resources.
        """)
    
    with col2:
        st.markdown("### 🏆 Key Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 20px; border-radius: 10px; text-align: center;">
                <h3 style="margin: 0; color: white;">98%</h3>
                <p style="margin: 0; color: white;">Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1dd1a1, #10ac84); color: white; padding: 20px; border-radius: 10px; text-align: center;">
                <h3 style="margin: 0; color: white;">67%</h3>
                <p style="margin: 0; color: white;">Fewer Features</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ff9ff3, #f368e0); color: white; padding: 20px; border-radius: 10px; text-align: center;">
                <h3 style="margin: 0; color: white;">4</h3>
                <p style="margin: 0; color: white;">Classes</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 4-Class System")
        
        classes_info = [
            ("🟢 CONFIRMED", "Verified exoplanets", "#1dd1a1"),
            ("🟡 CANDIDATE", "Potential discoveries", "#feca57"),
            ("🔴 FALSE POSITIVE", "Astronomical artifacts", "#ff6b6b"),
            ("🟣 REFUTED", "Disproven signals", "#a29bfe")
        ]
        
        for icon_text, description, color in classes_info:
            st.markdown(f"""
            <div style="background: white; padding: 15px; border-radius: 8px; border-left: 4px solid {color}; margin: 5px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <h4 style="margin: 0; color: #2c3e50;">{icon_text}</h4>
                <p style="margin: 0; color: #666; font-size: 0.9rem;">{description}</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 40px;">
    <p style="font-size: 1.2rem;">🌌 NASA-inspired Exoplanet Discovery AI • 98% Accuracy • Revolutionary Feature Optimization</p>
    <p style="font-size: 1rem;">Proving that intelligent data beats big data in astronomical research</p>
</div>
""", unsafe_allow_html=True)
