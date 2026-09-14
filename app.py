import pickle
import numpy as np
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Concrete AI | Strength Predictor",
    page_icon="⚡",
    layout="wide",
)

# Custom Styling (Dark Glassmorphism + Glow Accents)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* App Background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(18, 24, 38) 0%, rgb(9, 13, 21) 90%);
        color: #f1f5f9;
    }

    /* Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px 24px;
        backdrop-filter: blur(12px);
        margin-bottom: 20px;
    }

    /* Primary Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
        color: #ffffff;
        border: none;
        border-radius: 12px;
        padding: 14px 24px;
        font-size: 16px;
        font-weight: 700;
        letter-spacing: 0.5px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 24px -6px rgba(14, 165, 233, 0.5);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px -4px rgba(99, 102, 241, 0.6);
    }

    /* Prediction Result Badge */
    .result-badge {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(5, 150, 105, 0.05));
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-top: 20px;
    }
    .result-val {
        font-size: 42px;
        font-weight: 800;
        color: #34d399;
        margin: 6px 0;
    }
    .result-label {
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: #94a3b8;
    }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("concrete_model.h5")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


model, scaler = load_assets()

# Header
st.markdown(
    """
    <div style="text-align: center; padding: 25px 0 35px 0;">
        <span style="background: rgba(14, 165, 233, 0.15); color: #38bdf8; padding: 6px 14px; border-radius: 50px; font-size: 12px; font-weight: 700; letter-spacing: 1px;">ANN REGRESSION ENGINE</span>
        <h1 style="font-size: 40px; font-weight: 800; margin: 12px 0 6px 0; background: linear-gradient(90deg, #f8fafc, #94a3b8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Concrete Compressive Strength</h1>
        <p style="color: #94a3b8; font-size: 15px; max-width: 600px; margin: 0 auto;">Enter mix parameters and curing duration to simulate structural compressive capacity in real time.</p>
    </div>
""",
    unsafe_allow_html=True,
)

# Main Grid Layout
left_col, right_col = st.columns([1.7, 1], gap="large")

with left_col:
    st.markdown("### 🧪 Mix Proportions")

    col1, col2 = st.columns(2)
    with col1:
        cement = st.number_input(
            "Cement (kg/m³)", 0.0, 600.0, 319.0, step=5.0
        )
        blast_furnace_slag = st.number_input(
            "Blast Furnace Slag (kg/m³)", 0.0, 400.0, 181.0, step=5.0
        )
        fly_ash = st.number_input(
            "Fly Ash (kg/m³)", 0.0, 300.0, 99.0, step=5.0
        )
        water = st.number_input(
            "Water (kg/m³)", 50.0, 300.0, 184.0, step=2.0
        )

    with col2:
        superplasticizer = st.number_input(
            "Superplasticizer (kg/m³)", 0.0, 40.0, 15.0, step=0.5
        )
        coarse_agg = st.number_input(
            "Coarse Aggregate (kg/m³)", 600.0, 1300.0, 967.0, step=10.0
        )
        fine_agg = st.number_input(
            "Fine Aggregate (kg/m³)", 400.0, 1100.0, 791.0, step=10.0
        )
        age = st.slider("Curing Age (Days)", 1, 365, 28)

with right_col:
    st.markdown("### 📊 Mix Summary")

    total_binder = cement + blast_furnace_slag + fly_ash
    water_binder_ratio = (
        round(water / total_binder, 2) if total_binder > 0 else 0.0
    )

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="result-label">Total Binder Content</div>
            <div style="font-size: 24px; font-weight: 700; color: #f8fafc; margin-top: 4px;">{total_binder:.1f} <span style="font-size: 14px; color: #64748b;">kg/m³</span></div>
            <hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.06); margin: 14px 0;">
            <div class="result-label">Water / Binder Ratio</div>
            <div style="font-size: 24px; font-weight: 700; color: #38bdf8; margin-top: 4px;">{water_binder_ratio}</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    if st.button("Calculate Strength"):
        inputs = np.array(
            [
                [
                    cement,
                    blast_furnace_slag,
                    fly_ash,
                    water,
                    superplasticizer,
                    coarse_agg,
                    fine_agg,
                    age,
                ]
            ]
        )

        scaled = scaler.transform(inputs)
        raw_pred = model.predict(scaled)
        strength = max(0.0, float(raw_pred[0][0]))

        st.markdown(
            f"""
            <div class="result-badge">
                <div class="result-label">Estimated 28-Day Equivalent</div>
                <div class="result-val">{strength:.2f} <span style="font-size: 20px; font-weight: 600;">MPa</span></div>
                <p style="color: #94a3b8; font-size: 13px; margin: 0;">Predicted via Multi-Layer Perceptron ANN</p>
            </div>
        """,
            unsafe_allow_html=True,
        )