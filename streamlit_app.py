# streamlit_app.py — Member 3: Streamlit Frontend
# Run with: streamlit run streamlit_app.py
# Make sure FastAPI backend is running first: uvicorn app:app --reload --port 8000

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CropSense — GNN Crop Recommender",
    page_icon="🌾",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0b0f0a; }
    [data-testid="stSidebar"]          { background-color: #0f1710; }

    /* All general text bright white */
    p, span, div, label                { color: #e8f0e6 !important; }

    /* Headings green */
    h1, h2, h3                         { color: #4ade80 !important; }

    /* Metric labels and values */
    [data-testid="stMetricLabel"]      { color: #a3c49e !important; font-size: 0.9rem !important; }
    [data-testid="stMetricValue"]      { color: #ffffff !important; font-size: 1.8rem !important; }
    [data-testid="stMetricDelta"]      { color: #fbbf24 !important; }

    /* Slider labels and numbers */
    [data-testid="stSlider"] label     { color: #e8f0e6 !important; }
    [data-testid="stSlider"] p         { color: #e8f0e6 !important; }
    .stSlider span                     { color: #a3c49e !important; }

    /* Caption and markdown text */
    [data-testid="stMarkdown"] p       { color: #e8f0e6 !important; }
    .stCaption                         { color: #a3c49e !important; }

    /* Form labels */
    [data-testid="stNumberInput"] label { color: #a3c49e !important; }
    [data-testid="stForm"] label        { color: #a3c49e !important; }

    /* Info / warning / success boxes */
    [data-testid="stAlert"] p          { color: #0b0f0a !important; }

    /* Button */
    .stButton > button {
        background-color: #4ade80;
        color: #0b0f0a;
        font-weight: 700;
        border: none;
        border-radius: 8px;
    }
    .stButton > button:hover { background-color: #86efac; color: #0b0f0a; }

    /* Result card */
    .result-card {
        background: #1a2317;
        border: 1px solid #4ade80;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Backend URL ───────────────────────────────────────────────────────────────
BACKEND = "http://localhost:8000"   # FastAPI runs on 8000 (not 5000)

CROP_EMOJIS = {
    "rice": "🌾", "wheat": "🌿", "maize": "🌽", "cotton": "🌸",
    "sugarcane": "🎋", "coffee": "☕", "jute": "🪢", "mango": "🥭",
    "banana": "🍌", "grapes": "🍇", "watermelon": "🍉", "apple": "🍎",
    "orange": "🍊", "papaya": "🍈", "coconut": "🥥", "pomegranate": "🍎",
    "lentil": "🫘", "blackgram": "🫘", "mungbean": "🫘", "mothbeans": "🫘",
    "pigeonpeas": "🫘", "kidneybeans": "🫘", "chickpea": "🫘",
}

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🌾 CropSense")
st.caption("GNN-powered crop recommendation · Enter soil and climate data to get started")
st.divider()

# ── Backend health check ──────────────────────────────────────────────────────
try:
    health = requests.get(f"{BACKEND}/health", timeout=3).json()
    st.success(f"✅ Backend connected · {health['num_crops']} crop classes loaded")
    with st.expander("View API docs"):
        st.markdown(f"Interactive API docs: [{BACKEND}/docs]({BACKEND}/docs)")
except Exception:
    st.warning("⚠️ FastAPI backend not running. Start it with: `uvicorn app:app --reload --port 8000`")

st.divider()

# ── Layout ────────────────────────────────────────────────────────────────────
input_col, result_col = st.columns([1, 1.3], gap="large")

# ════════════════════════════════════════════
#  INPUT FORM
# ════════════════════════════════════════════
with input_col:
    st.subheader("🧪 Soil & Climate Input")

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)

        with c1:
            N           = st.number_input("N — Nitrogen (kg/ha)",   0.0,  140.0, 90.0,  1.0)
            K           = st.number_input("K — Potassium (kg/ha)",  5.0,  205.0, 43.0,  1.0)
            humidity    = st.number_input("Humidity (%)",           14.0, 100.0, 82.0,  0.1)
            rainfall    = st.number_input("Rainfall (mm)",          20.0, 300.0, 202.9, 0.1)

        with c2:
            P           = st.number_input("P — Phosphorus (kg/ha)", 5.0,  145.0, 42.0,  1.0)
            temperature = st.number_input("Temperature (°C)",       8.0,  44.0,  20.8,  0.1)
            ph          = st.number_input("pH Level (0–14)",        3.5,  9.9,   6.5,   0.1)

        predict_clicked = st.form_submit_button(
            "🚀 Run GNN Prediction", use_container_width=True, type="primary"
        )

    robustness_clicked = st.button(
        "⚡ Run Robustness Test (Normal vs Adversarial)",
        use_container_width=True
    )

# ════════════════════════════════════════════
#  RESULTS
# ════════════════════════════════════════════
with result_col:
    st.subheader("📊 Prediction Results")

    payload = {
        "N": N, "P": P, "K": K,
        "temperature": temperature,
        "humidity":    humidity,
        "ph":          ph,
        "rainfall":    rainfall
    }

    # ── Run prediction ──
    if predict_clicked:
        with st.spinner("Running GNN inference..."):
            try:
                resp   = requests.post(f"{BACKEND}/predict", json=payload, timeout=15)
                result = resp.json()

                if resp.status_code == 200:
                    st.session_state["prediction"] = result
                else:
                    st.error(f"❌ {result.get('detail', 'Unknown error')}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot reach FastAPI backend. Run `uvicorn app:app --reload --port 8000` first.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")

    # ── Show prediction ──
    if "prediction" in st.session_state:
        r          = st.session_state["prediction"]
        crop       = r["crop"]
        confidence = r["confidence"]
        top3       = r.get("top3", [])
        emoji      = CROP_EMOJIS.get(crop.lower(), "🌱")

        # Result card
        st.markdown(f"""
        <div class="result-card">
            <div style="font-size:3.5rem">{emoji}</div>
            <div style="font-size:2rem;font-weight:800;color:#4ade80;
                        text-transform:capitalize;margin:0.3rem 0">{crop}</div>
            <div style="color:#7a9275;font-size:0.85rem">Recommended Crop</div>
        </div>
        """, unsafe_allow_html=True)

        st.metric("Confidence Score", f"{confidence * 100:.1f}%")
        st.progress(float(confidence))

        if top3:
            st.markdown("**Top 3 Candidates**")
            df_top3 = pd.DataFrame(top3)
            fig = px.bar(
                df_top3, x="prob", y="crop", orientation="h",
                color="prob",
                color_continuous_scale=["#1a2317", "#4ade80"],
                text=df_top3["prob"].apply(lambda x: f"{x*100:.1f}%"),
                labels={"prob": "Probability", "crop": "Crop"},
            )
            fig.update_layout(
                paper_bgcolor="#0b0f0a", plot_bgcolor="#131a11",
                font_color="#e8f0e6",   coloraxis_showscale=False,
                margin=dict(l=0, r=10, t=10, b=0), height=200,
                xaxis=dict(gridcolor="#2a3828", range=[0, 1]),
                yaxis=dict(gridcolor="#2a3828"),
            )
            fig.update_traces(textposition="outside", marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👈 Fill in the soil data and click **Run GNN Prediction**")


# ════════════════════════════════════════════
#  ROBUSTNESS TEST (full width)
# ════════════════════════════════════════════
if robustness_clicked:
    st.divider()
    st.subheader("⚡ Robustness Test — Normal vs Adversarial (FGSM)")

    epsilon = st.slider("Epsilon (noise strength)", 0.01, 0.5, 0.1, 0.01)

    with st.spinner("Running adversarial robustness test..."):
        try:
            resp = requests.post(
                f"{BACKEND}/robustness-test",
                json={"input": payload, "epsilon": epsilon},
                timeout=15
            )
            rob = resp.json()

            if resp.status_code == 200:
                st.session_state["robustness"] = rob
            else:
                st.error(f"❌ {rob.get('detail', 'Unknown error')}")

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot reach FastAPI backend.")
        except Exception as e:
            st.error(f"❌ {e}")

if "robustness" in st.session_state:
    rob    = st.session_state["robustness"]
    n_conf = rob["normal"]["confidence"]
    a_conf = rob["adversarial"]["confidence"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Normal Prediction",
            rob["normal"]["crop"].capitalize(),
            f"{n_conf * 100:.1f}% confidence"
        )
    with col2:
        st.metric(
            "Adversarial Prediction",
            rob["adversarial"]["crop"].capitalize(),
            f"{a_conf * 100:.1f}% confidence",
            delta_color="inverse"
        )
    with col3:
        same = rob["same_prediction"]
        st.metric(
            "Prediction Stable?",
            "✅ Yes" if same else "❌ Changed",
            f"Confidence drop: {rob['confidence_drop'] * 100:.1f}%",
            delta_color="inverse"
        )

    # Grouped bar chart
    st.markdown("**Confidence: Normal vs Adversarial**")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        name="Normal", x=["Confidence"], y=[n_conf * 100],
        marker_color="#4ade80",
        text=[f"{n_conf*100:.1f}%"], textposition="outside"
    ))
    fig2.add_trace(go.Bar(
        name="Adversarial (FGSM)", x=["Confidence"], y=[a_conf * 100],
        marker_color="#fbbf24",
        text=[f"{a_conf*100:.1f}%"], textposition="outside"
    ))
    fig2.update_layout(
        barmode="group",
        paper_bgcolor="#0b0f0a", plot_bgcolor="#131a11",
        font_color="#e8f0e6",
        legend=dict(bgcolor="#131a11", bordercolor="#2a3828"),
        yaxis=dict(gridcolor="#2a3828", range=[0, 115], title="Confidence (%)"),
        margin=dict(l=0, r=0, t=20, b=0),
        height=300,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("🔗 FastAPI backend: `localhost:8000` · API docs: `localhost:8000/docs` · Model: GCN (Member 2) · Preprocessing: Member 1 · UI/Integration: Member 3")