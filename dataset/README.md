# AI Predictive Maintenance & Fault Detection Dashboard

## Overview
This project implements an AI-based predictive maintenance system designed to detect potential machine failures using industrial sensor data. The system analyzes parameters such as air temperature, process temperature, rotational speed, torque, and tool wear to predict machine faults and estimate machine health.

An interactive dashboard built using Streamlit visualizes machine conditions, failure risk, and performance metrics in real time.

---

## Key Features
- Machine failure prediction using Machine Learning
- Interactive dashboard for machine health monitoring
- Real-time input of sensor parameters
- Failure risk analysis and visualization
- Predictive insights for preventive maintenance

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Data Visualization

---

## Project Structure
AI_Predictive_Fault_Detection
│
├── app.py
├── dataset
│ └── ai4i2020.csv
└── README.md

---

## Dataset
The project uses the **AI4I 2020 Predictive Maintenance Dataset**, which contains simulated industrial machine sensor data used to predict machine failures.

Parameters include:
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear
- Machine Type

---

## How to Run the Project

### 1 Install Dependencies
pip install -r requirements.txt
### 2 Run the Dashboard
streamlit run dataset/app.py

The dashboard will open in your browser where you can enter machine parameters and predict failure risk.

---

## Applications
- Industrial predictive maintenance
- Smart manufacturing systems
- Machine health monitoring
- Failure risk analysis
- Industry 4.0 solutions

---

## Future Enhancements
- Real-time IoT sensor integration
- Remaining Useful Life (RUL) prediction
- Live machine monitoring dashboard
- Advanced anomaly detection

---

## Author
Devi Swathi