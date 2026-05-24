import streamlit as st
import pickle
import numpy as np
import pandas as pd

# 1. मॉडल और डेटा लोड करें
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# पेज की सेटिंग
st.set_page_config(page_title="Laptop Predictor", page_icon="💻", layout="wide")

# बैकग्राउंड इमेज URL (एक हाई-क्वालिटी लैपटॉप ड्राइंग)
bg_img = "https://images.unsplash.com/photo-1517694712202-14dd9538aa97?ixlib=rb-1.2.1&auto=format&fit=crop&w=1920&q=80"

# CSS: पूरी ऐप को डार्क और ग्लास जैसा बनाने के लिए
st.markdown(f"""
    <style>
    /* बैकग्राउंड इमेज सेटअप */
    .stApp {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.8)), url("{bg_img}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    /* सफ़ेद बैकग्राउंड हटाना और ग्लास लुक देना */
    div[data-testid="stVerticalBlock"] > div:has(div.main-box) {{
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
    }}

    /* इनपुट बॉक्स को डार्क करना (White background removal) */
    .stSelectbox div[data-baseweb="select"], .stNumberInput div[data-baseweb="input"] {{
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }}

    /* इनपुट टेक्स्ट का रंग सफ़ेद */
    div[data-baseweb="select"] *, input {{
        color: white !important;
    }}

    /* लेबल (Labels) का रंग नियॉन ब्लू */
    label {{
        color: #00d4ff !important;
        font-weight: bold !important;
        text-transform: uppercase;
    }}

    /* प्रेडिक्ट बटन स्टाइल */
    .stButton>button {{
        width: 100%;
        background: linear-gradient(90deg, #00d4ff, #0055ff);
        color: white;
        font-weight: bold;
        font-size: 20px;
        border: none;
        padding: 12px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
    }}
    </style>
    """, unsafe_allow_html=True)

st.title("💻 AI Laptop Price Predictor")
st.markdown("##### Fill in the specifications to get the estimated market price")

# फॉर्म कंटेनर
with st.container():
    st.markdown('<div class="main-box">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        company = st.selectbox('Brand', df['Company'].unique())
        type = st.selectbox('Type', df['TypeName'].unique())
        ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
        weight = st.number_input('Weight (kg)', min_value=0.5, value=2.0)

    with col2:
        touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
        ips = st.selectbox('IPS Panel', ['No', 'Yes'])
        screen_size = st.number_input('Screen Size (Inches)', min_value=10.0, value=15.6)
        resolution = st.selectbox('Resolution', ['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2560x1600','2560x1440'])

    st.markdown("---")
    col3, col4, col5 = st.columns(3)
    with col3:
        cpu = st.selectbox('CPU', df['Cpu_brand'].unique())
    with col4:
        gpu = st.selectbox('GPU', df['Gpu_brand'].unique())
    with col5:
        os = st.selectbox('OS', df['os'].unique())

    col6, col7 = st.columns(2)
    with col6:
        hdd = st.selectbox('HDD (GB)', [0, 128, 256, 512, 1024, 2048])
    with col7:
        ssd = st.selectbox('SSD (GB)', [0, 8, 128, 256, 512, 1024])

    st.markdown('</div>', unsafe_allow_html=True)

# Prediction Logic
if st.button('🚀 PREDICT PRICE'):
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

    query_df = pd.DataFrame([[
        company, type, ram, weight, 
        1 if touchscreen == 'Yes' else 0, 1 if ips == 'Yes' else 0, 
        ppi, cpu, hdd, ssd, gpu, os
    ]], columns=['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'Ips', 'ppi', 'Cpu_brand', 'HDD', 'SSD', 'Gpu_brand', 'os'])

    prediction = np.exp(pipe.predict(query_df)[0])

    st.balloons()
    st.markdown(f"""
        <div style="background: rgba(0, 212, 255, 0.1); padding: 20px; border-radius: 15px; border: 2px solid #00d4ff; text-align: center; margin-top: 20px;">
            <h2 style="color: white;">🎯 Predicted Price</h2>
            <h1 style="color: #00d4ff;">€ {int(prediction)}</h1>
            <h3 style="color: #f1f1f1;">Approx: ₹ {int(prediction * 90)}</h3>
        </div>
    """, unsafe_allow_html=True)