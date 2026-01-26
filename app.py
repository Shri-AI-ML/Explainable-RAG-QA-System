import streamlit as st
import pandas as pd
import numpy as np

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="DataForge",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CSS =================
st.markdown("""
<style>

/* Hide Streamlit defaults */
#MainMenu, footer, header {visibility: hidden;}

/* App background */
.stApp {
    background:
        radial-gradient(circle at 20% 10%, #2a004f, transparent 40%),
        radial-gradient(circle at 80% 80%, #3b0a6b, transparent 40%),
        linear-gradient(180deg, #020617, #0b0220);
    color: white;
    font-family: 'Inter', sans-serif;
}

/* Container spacing */
.block-container {
    padding-top: 3rem;
    max-width: 1400px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #12002a);
    border-right: 1px solid rgba(255,255,255,0.1);
}
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Hero heading */
.hero {
    font-size: 72px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #ff5ec4, #b86bff, #3cf2ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 30px rgba(255, 94, 196, 0.35);
    margin-bottom: 8px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 20px;
    color: #cbd5ff;
    margin-bottom: 70px;
}

/* Glass card */
.card {
    background: linear-gradient(
        180deg,
        rgba(255,255,255,0.10),
        rgba(255,255,255,0.03)
    );
    backdrop-filter: blur(20px);
    border-radius: 22px;
    padding: 30px 32px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow:
        0 0 40px rgba(186, 104, 255, 0.15),
        inset 0 0 20px rgba(255,255,255,0.05);
    transition: all 0.3s ease;
    height: 100%;
}

.card:hover {
    transform: translateY(-6px);
    box-shadow:
        0 0 60px rgba(255, 94, 196, 0.25),
        0 0 100px rgba(59, 242, 255, 0.15);
}

/* Card headings */
.card h3 {
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 14px;
}

/* Card text */
.card p {
    font-size: 17px;
    line-height: 1.6;
    color: #dbeafe;
}

</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.markdown(
    "<h2 style='background:linear-gradient(90deg,#ff5ec4,#b86bff,#3cf2ff);"
    "-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>"
    "DataForge</h2>",
    unsafe_allow_html=True
)
st.sidebar.caption("From raw data to decisions")

page = st.sidebar.radio("Navigate", ["Home", "EDA", "ML Demo", "About"])

# ================= HOME =================
if page == "Home":
    st.markdown("<div class='hero'>DataForge</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>A unified platform for data exploration & prediction</div>",
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.markdown("""
        <div class="card">
            <h3>📊 Smart EDA</h3>
            <p>Instant insights, distributions and data health checks.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
            <h3>🧠 ML Predictions</h3>
            <p>Confidence-driven predictions for faster decisions.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="card">
            <h3>🚀 Deploy Ready</h3>
            <p>Clean UI, demo-safe, and production deployable.</p>
        </div>
        """, unsafe_allow_html=True)

# ================= EDA =================
elif page == "EDA":
    st.markdown("<div class='hero' style='font-size:48px;'>EDA Overview</div>", unsafe_allow_html=True)

    df = pd.DataFrame({
        "Feature A": np.random.normal(50, 10, 300),
        "Feature B": np.random.normal(30, 6, 300),
        "Target": np.random.choice([0, 1], 300)
    })

    m1, m2, m3 = st.columns(3)
    m1.metric("Rows", df.shape[0])
    m2.metric("Features", df.shape[1])
    m3.metric("Missing Values", 0)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Feature A Distribution")
        st.bar_chart(df["Feature A"])
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Target Distribution")
        st.bar_chart(df["Target"].value_counts())
        st.markdown("</div>", unsafe_allow_html=True)

# ================= ML DEMO =================
elif page == "ML Demo":
    st.markdown("<div class='hero' style='font-size:48px;'>ML Prediction Demo</div>", unsafe_allow_html=True)

    left, right = st.columns([1, 2])

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Input Parameters")
        a = st.slider("Feature A", 0, 100, 55)
        b = st.slider("Feature B", 0, 60, 30)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Model Output")

        score = min(0.95, 0.45 + (a + b) / 180)
        result = "Positive" if score > 0.6 else "Negative"

        st.metric("Prediction", result)
        st.progress(score)
        st.caption(f"Confidence Score: {round(score*100, 1)}%")

        st.markdown("</div>", unsafe_allow_html=True)

# ================= ABOUT =================
else:
    st.markdown("<div class='hero' style='font-size:48px;'>About DataForge</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <b>Problem</b><br>
        Raw data is hard to interpret quickly.

        <br><br><b>Solution</b><br>
        DataForge unifies EDA and ML predictions into one interface.

        <br><br><b>Tech Stack</b><br>
        Python · Streamlit · Pandas · NumPy · ML Models
    </div>
    """, unsafe_allow_html=True)

    st.success("Demo ready. Deployment safe. 😌")
