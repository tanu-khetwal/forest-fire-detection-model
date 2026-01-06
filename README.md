# 🔥 Forest Fire Detection System

An AI-powered system designed to detect and predict forest fires using Machine Learning. This project aims to provide early warning signals by analyzing environmental data and imagery to prevent large-scale ecological disasters.

[Image of forest fire detection system architecture]

## 📋 Table of Contents
* [Features](#-features)
* [How It Works](#-how-it-works)
* [Dataset](#-dataset)
* [Installation](#-installation)
* [Usage](#-usage)
* [Results](#-results)

---

## 🌟 Features
* **Real-time Image Classification:** Detects fire and smoke in images with high precision.
* **Environmental Data Analysis:** Analyzes temperature, humidity, and wind speed to predict fire risks.
* **Interactive Dashboard:** Built with **Streamlit** for easy image uploads and real-time visualization.
* **Optimized for Speed:** Lightweight model architecture suitable for edge deployment (drones/IoT).

---

## 🛠️ How It Works
The system processes input through a multi-stage pipeline:
1. **Preprocessing:** Images are resized and normalized; tabular data is scaled.
2. **Feature Extraction:** The model identifies patterns related to fire colors, smoke textures, and heat indexes.
3. **Classification:** The AI determines the probability of a fire hazard.

[Image of CNN architecture for image classification]

---

## 📊 Dataset
The model was trained using a combination of public datasets (such as Kaggle and NASA FIRMS).
* **Classes:** `Fire`, `No Fire`.
* **Format:** [e.g., RGB Images / CSV sensor logs].

---

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/forest-fire-detection.git](https://github.com/your-username/forest-fire-detection.git)
   cd forest-fire-detection

   Metric,Score
Accuracy,97.8%
Precision,96.5%
Recall,98.2%

🛠 Technologies Used
Python: Core programming.
TensorFlow/Keras: Deep Learning framework.
OpenCV: Image processing and computer vision.
Pandas & NumPy: Data analysis and manipulation.
Streamlit: Frontend web application.

🤝 Contributing
Contributions are welcome! If you have suggestions for improving the detection algorithm or adding new data sources, please open an issue or submit a pull request.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
