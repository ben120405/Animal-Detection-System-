 🐾 Animal Detection System

An AI-powered Animal Detection dashboard that detects animals from images, videos, and live camera feeds using a trained **YOLOv8** model. The system highlights carnivorous animals, stores detection results in a database, and visualizes detection analytics through an interactive **Streamlit** dashboard.

 📌 Project Overview

This project demonstrates how deep learning can be used to monitor wildlife and detect animal species in real time. The system can:

- ✅ Detect multiple animals in images and videos
- 🔴 Highlight carnivorous animals (lion and tiger)
- 💾 Store detection results in a database
- 📊 Display detection analytics and history
- 🖥️ Provide an interactive AI dashboard

🚀 Features

🧠 AI Animal Detection
- Uses a trained YOLOv8 model to detect animal species from images and videos

🔴 Carnivore Detection
- Carnivorous animals (lion, tiger) highlighted with **red bounding boxes**

📊 Detection Dashboard
Displays:
- Detection history table
- Animal detection counts
- Interactive analytics charts

💾 Database Storage
Every detection stored in SQLite database:
- Animal type
- Confidence score
- Timestamp

📷 Multiple Input Modes
- 🖼️ Image upload detection
- 🎥 Video detection
- 📷 Live webcam detection

🧰 Technologies Used

| Technology | Purpose |
|------------|---------|
| [Python](https://python.org) | Backend programming |
| [Streamlit](https://streamlit.io) | Web dashboard |
| [YOLOv8](https://ultralytics.com) | Object detection model |
| [OpenCV](https://opencv.org) | Image/video processing |
| [SQLite](https://sqlite.org) | Detection database |
| [Plotly](https://plotly.com) | Data visualization |
| [Pandas](https://pandas.pydata.org) | Data analysis |

🏗️ Project Structure

animal-detection-system/
│
├── app.py # Main Streamlit application
├── database.py # SQLite database operations
├── requirements.txt # Python dependencies
├── runtime.txt # Heroku runtime (optional)
│
├── models/
│ └── animal_v2/
│ └── weights/
│ └── best.pt # Trained YOLOv8 model
│
├── inference/ # Inference scripts (optional)
│
└── .streamlit/
└── config.toml # Streamlit configuration

text

⚙️ Installation

1️⃣ Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/animal-detection-system.git
cd animal-detection-system
2️⃣ Create virtual environment
bash
python -m venv venv
Activate it:

Windows: venv\Scripts\activate

Linux/Mac: source venv/bin/activate

3️⃣ Install dependencies
bash
pip install -r requirements.txt
▶️ Run the Application
bash
streamlit run app.py
The dashboard will automatically open in your browser at http://localhost:8501

📊 Dashboard Pages
Page	Description
Dashboard	System statistics, detection history, analytics
Image Detection	Upload images for animal detection
Video Detection	Process uploaded videos frame-by-frame
Camera Mode	Live webcam detection
Model Info	Model details and class information
📈 Detection Analytics
The dashboard displays:

Total detections

Animal detection counts

Carnivore alerts

Interactive bar charts

Example output:

text
🦁 Lion: 5 detections
🐅 Tiger: 3 detections
🐘 Elephant: 7 detections
🦓 Zebra: 4 detections
🧠 Model Details
Property	Value
Model	YOLOv8
Framework	Ultralytics
Detected Classes	Buffalo, Elephant, Rhino, Zebra, Lion, Tiger
🌐 Deployment
Streamlit Community Cloud (Recommended)
Upload project to GitHub

Connect repository on Streamlit Cloud

Deploy using app.py

Other Options
Heroku

Railway

Render

📌 Future Improvements
🌍 Real-time wildlife monitoring system

🔥 AI detection heatmaps

📄 Automatic PDF detection reports

☁️ Cloud database integration

📱 Edge device deployment (Raspberry Pi)

🤝 Contributing
Fork the project

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

📜 License
This project is intended for educational and research purposes.

text
MIT License

Copyright (c) 2026 [Benin Dbritto]

Permission is hereby granted, free of charge...
👨‍💻 Author
Benin Dbritto

Connect with me:
GitHub:https://github.com/ben120405
LinkedIn:https://www.linkedin.com/in/benin-dbritto-2a27a9332/

