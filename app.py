import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Animal Detection System",
    page_icon="🐾",
    layout="wide"
)
# ---------------- SETTINGS ----------------
MODEL_PATH = "models/animal_v2/weights/best.pt"
CARNIVORES = {"lion", "tiger"}

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ---------------- CUSTOM CSS ----------------
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #0e1117;
        padding: 16px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #262730;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .metric-label {
        font-size: 14px;
        color: #a3a3a3;
    }
    .section-title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- SIDEBAR NAVIGATION ----------------
with st.sidebar:
    selected = option_menu(
        " Animal Detection System",
        ["Dashboard", "Image Detection", "Video Detection", "Camera Mode", "Model Info"],
        icons=["speedometer", "image", "camera-video", "camera", "cpu"],
        menu_icon="robot",
        default_index=0,
    )

# =====================================================
# DASHBOARD
# =====================================================
if selected == "Dashboard":
    st.markdown(
        "<p class='section-title'>Animal Detection Dashboard</p>",
        unsafe_allow_html=True,
    )

    st.write("AI system for detecting animals and identifying carnivores.")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">YOLOv8</div>
                <div class="metric-label">Model Type</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{len(model.names)}</div>
                <div class="metric-label">Total Classes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">2</div>
                <div class="metric-label">Carnivore Classes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.subheader("Animal Classes")

    df = pd.DataFrame(
        {
            "Class": list(model.names.values()),
        }
    )
    st.dataframe(df, use_container_width=True)

    st.subheader("Dataset Distribution Example")

    # Create a sample distribution list matching number of classes
    base_samples = [50, 45, 48, 40, 30, 25]
    samples = (base_samples * ((len(model.names) // len(base_samples)) + 1))[
        : len(model.names)
    ]

    chart_df = pd.DataFrame(
        {
            "Animal": list(model.names.values()),
            "Samples": samples,
        }
    )

    fig = px.bar(
        chart_df,
        x="Animal",
        y="Samples",
        color="Animal",
        title="Animal Dataset Distribution",
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# IMAGE DETECTION
# =====================================================
elif selected == "Image Detection":
    st.title("🖼 Image Detection")

    uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded is not None:
        # Save uploaded file to a temporary location
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tfile.write(uploaded.read())
        temp_path = tfile.name

        # Run prediction
        results = model.predict(temp_path, conf=0.4, verbose=False)
        frame = cv2.imread(temp_path)

        carnivore_count = 0
        animals = []

        if results and len(results) > 0:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if label.lower() in CARNIVORES:
                    color = (0, 0, 255)  # Red for carnivores
                    carnivore_count += 1
                else:
                    color = (255, 0, 0)  # Blue for others

                animals.append(label)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

        left, right = st.columns([2, 1])

        left.image(frame, channels="BGR")

        with right:
            st.subheader("Detection Summary")

            st.metric("Carnivores Detected", carnivore_count)

            if carnivore_count > 0:
                st.error("⚠ Carnivore Detected!")

            if animals:
                chart_df = pd.DataFrame({"Animal": animals})
                fig = px.histogram(chart_df, x="Animal", color="Animal")
                st.plotly_chart(fig, use_container_width=True)

# =====================================================
# VIDEO DETECTION
# =====================================================
elif selected == "Video Detection":
    st.title("🎥 Video Detection")

    uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())

        cap = cv2.VideoCapture(tfile.name)

        frame_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            results = model.predict(frame, conf=0.4, verbose=False)

            carnivore_count = 0

            if results and len(results) > 0:
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    conf = float(box.conf[0])

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if label.lower() in CARNIVORES:
                        color = (0, 0, 255)
                        carnivore_count += 1
                    else:
                        color = (255, 0, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    cv2.putText(
                        frame,
                        f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

            cv2.putText(
                frame,
                f"Carnivores: {carnivore_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3,
            )

            frame_placeholder.image(frame, channels="BGR")

        cap.release()

# =====================================================
# CAMERA MODE
# =====================================================
elif selected == "Camera Mode":
    st.title("📷 Live Camera Detection")

    run = st.checkbox("Start Camera")

    frame_window = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()

        if not ret:
            st.warning("Failed to access the camera.")
            break

        results = model.predict(frame, conf=0.4, verbose=False)

        if results and len(results) > 0:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if label.lower() in CARNIVORES:
                    color = (0, 0, 255)
                else:
                    color = (255, 0, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

        frame_window.image(frame, channels="BGR")

    cap.release()

# =====================================================
# MODEL INFO
# =====================================================
elif selected == "Model Info":
    st.title("🧠 Model Information")

    st.subheader("AI System Status")

    col1, col2 = st.columns(2)

    with col1:
        st.success("Model Loaded Successfully")

    with col2:
        st.info("Detection Engine Active")

    st.markdown("---")

    st.write("Model used: YOLOv8")

    st.write("Classes detected:")

    st.json(model.names)

    st.markdown(
        """
        **Carnivore Classes**

        - Lion
        - Tiger
        """
    )
