# streamlit_app.py — Member 3: Streamlit Frontend
# Run with: streamlit run streamlit_app.py
# Make sure FastAPI backend is running first: uvicorn app:app --reload --port 8000
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Hide only sklearn version warning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Hide PyTorch Geometric warnings
warnings.filterwarnings("ignore", message="An issue occurred while importing")
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CropSense",
    page_icon="🌾",
    layout="wide",
)

# ── Styling (Earthy & Organic Theme) ──────────────────────────────────────────
st.markdown("""
<style>
    /* Warm cream/beige background for the main app */
    [data-testid="stAppViewContainer"] { background-color: #F9F6F0; }
    [data-testid="stSidebar"]          { background-color: #E6E1D1; }
    [data-testid="stHeader"]           { background-color: rgba(249, 246, 240, 0.8); }

    /* All general text warm dark grey/brown */
    p, span, div, label                { color: #3E3E3E !important; }

    /* Headings deep forest green */
    h1, h2, h3                         { color: #2B5742 !important; font-family: 'Georgia', serif; }

    /* Metric labels and values */
    [data-testid="stMetricLabel"]      { color: #7BB07F !important; font-size: 0.9rem !important; font-weight: 600; }
    [data-testid="stMetricValue"]      { color: #2B5742 !important; font-size: 1.8rem !important; }
    [data-testid="stMetricDelta"]      { color: #C3604C !important; }

    /* Slider labels and numbers */
    [data-testid="stSlider"] label     { color: #3E3E3E !important; }
    [data-testid="stSlider"] p         { color: #3E3E3E !important; }
    .stSlider span                     { color: #2B5742 !important; }

    /* Caption and markdown text */
    [data-testid="stMarkdown"] p       { color: #3E3E3E !important; }
    .stCaption                         { color: #7BB07F !important; }

    /* Form labels */
    [data-testid="stNumberInput"] label { color: #5A4D41 !important; font-weight: 500; }
    [data-testid="stForm"] label        { color: #5A4D41 !important; }

    /* Input box styling */
    div[data-baseweb="input"] {
        border-radius: 12px !important;
        border: 1px solid #7BB07F !important;
        background-color: #FFFFFF !important;
    }

    /* Info / warning / success boxes */
    [data-testid="stAlert"] { background-color: #E6E1D1 !important; border: none !important; }
    [data-testid="stAlert"] p { color: #2B5742 !important; }

    /* Button (Terracotta Red/Brown) */
    .stButton > button {
        background-color: #C3604C;
        color: #FFFFFF !important;
        font-weight: 700;
        border: none;
        border-radius: 25px; /* Pill-shaped */
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover { background-color: #A34E3C; color: #FFFFFF !important; }
    .stButton > button * { color: #FFFFFF !important; } /* Ensure text inside stays white */

    /* Result card (Deep Forest Green to stand out) */
    .result-card {
        background: #89C992;
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    /* Force text inside the result card to be light so it's readable on green */
    .result-card div {
        color: #F9F6F0 !important; 
    }
</style>
""", unsafe_allow_html=True)

# ── Backend URL ───────────────────────────────────────────────────────────────
BACKEND = "http://localhost:8000"   # FastAPI runs on 8000

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
            N           = st.number_input("Nitrogen-kg/ha (0-140)",   0.0,  140.0, 90.0,  1.0)
            K           = st.number_input("Potassium-kg/ha (5-205)",  5.0,  205.0, 43.0,  1.0)
            humidity    = st.number_input("Humidity% (14-100)",       14.0, 100.0, 82.0,  0.1)
            rainfall    = st.number_input("Rainfall-mm (20-300)",     20.0, 300.0, 202.9, 0.1)

        with c2:
            P           = st.number_input("Phosphorus-kg/ha (5-145)", 5.0,  145.0, 42.0,  1.0)
            temperature = st.number_input("Temperature-°C (8-44)",    8.0,  44.0,  20.8,  0.1)
            ph          = st.number_input("pH level (3.5-9.9)",       3.5,  9.9,   6.5,   0.1)

        predict_clicked = st.form_submit_button(
            "🚀 Run GNN Prediction", use_container_width=True, type="primary"
        )

    # robustness_clicked = st.button(
    #     "⚡ Run Robustness Test (Normal vs Adversarial)",
    #     use_container_width=True
    # )

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

        # Result card - Updated colors for organic theme
        st.markdown(f"""
        <div class="result-card">
            <div style="font-size:3.5rem">{emoji}</div>
            <div style="font-size:2rem;font-weight:800;color:#F9F6F0 !important;
                        text-transform:capitalize;margin:0.3rem 0">{crop}</div>
            <div style="color:#7BB07F !important;font-size:0.85rem">Recommended Crop</div>
        </div>
        """, unsafe_allow_html=True)

        st.metric("Confidence Score", f"{confidence * 100:.1f}%")
        st.progress(float(confidence))

        if top3:
            st.markdown("**Top 3 Candidates**")
            df_top3 = pd.DataFrame(top3)
            # Updated Plotly colors for Earthy theme
            fig = px.bar(
                df_top3, x="prob", y="crop", orientation="h",
                color="prob",
                color_continuous_scale=["#E6E1D1", "#7BB07F"],
                text=df_top3["prob"].apply(lambda x: f"{x*100:.1f}%"),
                labels={"prob": "Probability", "crop": "Crop"},
            )
            fig.update_layout(
                paper_bgcolor="#F9F6F0", plot_bgcolor="#F9F6F0",
                font_color="#3E3E3E",   coloraxis_showscale=False,
                margin=dict(l=0, r=10, t=10, b=0), height=200,
                xaxis=dict(gridcolor="#E6E1D1", range=[0, 1]),
                yaxis=dict(gridcolor="#E6E1D1"),
            )
            fig.update_traces(textposition="outside", marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👈 Fill in the soil data and click **Run GNN Prediction**")


# ════════════════════════════════════════════
#  ROBUSTNESS TEST (full width)
# ════════════════════════════════════════════
# if robustness_clicked:
#     st.divider()
#     st.subheader("⚡ Robustness Test — Normal vs Adversarial (FGSM)")

#     epsilon = st.slider("Epsilon (noise strength)", 0.01, 0.5, 0.1, 0.01)

#     with st.spinner("Running adversarial robustness test..."):
#         try:
#             resp = requests.post(
#                 f"{BACKEND}/robustness-test",
#                 json={"input": payload, "epsilon": epsilon},
#                 timeout=15
#             )
#             rob = resp.json()

#             if resp.status_code == 200:
#                 st.session_state["robustness"] = rob
#             else:
#                 st.error(f"❌ {rob.get('detail', 'Unknown error')}")

#         except requests.exceptions.ConnectionError:
#             st.error("❌ Cannot reach FastAPI backend.")
#         except Exception as e:
#             st.error(f"❌ {e}")

# if "robustness" in st.session_state:
#     rob    = st.session_state["robustness"]
#     n_conf = rob["normal"]["confidence"]
#     a_conf = rob["adversarial"]["confidence"]

#     col1, col2, col3 = st.columns(3)

#     with col1:
#         st.metric(
#             "Normal Prediction",
#             rob["normal"]["crop"].capitalize(),
#             f"{n_conf * 100:.1f}% confidence"
#         )
#     with col2:
#         st.metric(
#             "Adversarial Prediction",
#             rob["adversarial"]["crop"].capitalize(),
#             f"{a_conf * 100:.1f}% confidence",
#             delta_color="inverse"
#         )
#     with col3:
#         same = rob["same_prediction"]
#         st.metric(
#             "Prediction Stable?",
#             "✅ Yes" if same else "❌ Changed",
#             f"Confidence drop: {rob['confidence_drop'] * 100:.1f}%",
#             delta_color="inverse"
#         )

    # Grouped bar chart - Updated Plotly colors
    # st.markdown("**Confidence: Normal vs Adversarial**")
    # fig2 = go.Figure()
    # fig2.add_trace(go.Bar(
    #     name="Normal", x=["Confidence"], y=[n_conf * 100],
    #     marker_color="#7BB07F", # Grass Green
    #     text=[f"{n_conf*100:.1f}%"], textposition="outside"
    # ))
    # fig2.add_trace(go.Bar(
    #     name="Adversarial (FGSM)", x=["Confidence"], y=[a_conf * 100],
    #     marker_color="#C3604C", # Terracotta Red
    #     text=[f"{a_conf*100:.1f}%"], textposition="outside"
    # ))
    # fig2.update_layout(
    #     barmode="group",
    #     paper_bgcolor="#F9F6F0", plot_bgcolor="#F9F6F0",
    #     font_color="#3E3E3E",
    #     legend=dict(bgcolor="#F9F6F0", bordercolor="#E6E1D1"),
    #     yaxis=dict(gridcolor="#E6E1D1", range=[0, 115], title="Confidence (%)"),
    #     margin=dict(l=0, r=0, t=20, b=0),
    #     height=300,
    # )
    # st.plotly_chart(fig2, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("🔗 FastAPI backend: `localhost:8000` · API docs: `localhost:8000/docs` · Model: GCN (Member 2) · Preprocessing: Member 1 · UI/Integration: Member 3")