import streamlit as st
import tempfile
import pandas as pd
import plotly.express as px
import cv2
from ultralytics import YOLO
from streamlit_option_menu import option_menu
from database import create_connection, create_table, insert_detection, get_all_detections

# IMPORTANT: Must be the first Streamlit command
st.set_page_config(
    page_title="Animal Detection System",
    page_icon="🐾",
    layout="wide"
)

# ---------------- DATABASE SETUP ----------------
conn = create_connection()
create_table(conn)

# ---------------- SETTINGS ----------------
MODEL_PATH = "models/animal_v2/weights/best.pt"
CARNIVORES = {"lion", "tiger"}

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    """Load the YOLO model with caching"""
    return YOLO(MODEL_PATH)

model = load_model()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    selected = option_menu(
        menu_title="Animal Detection System",
        options=["Dashboard", "Image Detection", "Video Detection", "Camera Mode", "Model Info"],
        icons=["speedometer", "image", "camera-video", "camera", "cpu"],
        menu_icon="robot",
        default_index=0,
    )

# ---------------- DASHBOARD ----------------
if selected == "Dashboard":
    st.title("🐾 Animal Detection Dashboard")
    
    st.write("AI system for detecting animals and identifying carnivores.")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Type", "YOLOv8")
    col2.metric("Classes", len(model.names))
    col3.metric("Carnivore Classes", 2)
    
    st.markdown("---")
    
    st.subheader("Animal Classes")
    df_classes = pd.DataFrame({
        "Class": list(model.names.values())
    })
    st.dataframe(df_classes, use_container_width=True)
    
    # Detection History
    st.subheader("Detection History")
    records = get_all_detections(conn)
    
    if records:
        df = pd.DataFrame(records, columns=["Animal", "Confidence", "Timestamp"])
        st.dataframe(df, use_container_width=True)
        
        # Analytics
        st.subheader("Animal Detection Counts")
        counts = df["Animal"].value_counts().reset_index()
        counts.columns = ["Animal", "Count"]
        
        fig = px.bar(
            counts,
            x="Animal",
            y="Count",
            color="Animal",
            title="Animal Detection Analytics"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No detections recorded yet.")

# ---------------- IMAGE DETECTION ----------------
elif selected == "Image Detection":
    st.title("🖼 Image Detection")
    
    uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    
    if uploaded:
        # Save uploaded image temporarily
        with open("temp.jpg", "wb") as f:
            f.write(uploaded.read())
        
        # Run prediction
        results = model.predict("temp.jpg", conf=0.4)
        frame = cv2.imread("temp.jpg")
        
        carnivore_count = 0
        animals = []
        
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
            
            # Save detection in database
            insert_detection(conn, label, conf)
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
        
        # Display results
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

# ---------------- VIDEO DETECTION ----------------
elif selected == "Video Detection":
    st.title("🎥 Video Detection")
    
    uploaded = st.file_uploader("Upload video", type=["mp4"])
    
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        tfile.close()
        
        cap = cv2.VideoCapture(tfile.name)
        frame_placeholder = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model.predict(frame, conf=0.4, verbose=False)
            carnivore_count = 0
            
            for box in results[0].boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                insert_detection(conn, label, conf)
                
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
                    2
                )
            
            # Add carnivore counter
            cv2.putText(
                frame,
                f"Carnivores: {carnivore_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )
            
            frame_placeholder.image(frame, channels="BGR")
        
        cap.release()

# ---------------- CAMERA MODE ----------------
elif selected == "Camera Mode":
    st.title("📷 Live Camera Detection")
    
    run = st.checkbox("Start Camera")
    frame_window = st.image([])
    
    cap = cv2.VideoCapture(0)
    
    while run:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.predict(frame, conf=0.4, verbose=False)
        
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            insert_detection(conn, label, conf)
            
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
                2
            )
        
        frame_window.image(frame, channels="BGR")
    
    cap.release()

# ---------------- MODEL INFO ----------------
elif selected == "Model Info":
    st.title("🧠 Model Information")
    
    col1, col2 = st.columns(2)
    col1.success("✅ Model Loaded Successfully")
    col2.info("🔥 Detection Engine Active")
    
    st.markdown("---")
    
    st.write("**Model used:** YOLOv8")
    st.write("**Classes detected:**")
    st.json(model.names)
    
    st.markdown("""
    **Carnivore Classes**
    - 🦁 Lion
    - 🐅 Tiger
    """)
