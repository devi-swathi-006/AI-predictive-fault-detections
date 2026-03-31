import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import streamlit.components.v1 as components

# ==========================
# PAGE SETTINGS
# ==========================
st.set_page_config(page_title="AI Fault Detection", layout="wide")

st.title("🔧 AI Predictive Fault Detection System")
st.markdown("""
<style>

.kpi-card {
background: linear-gradient(135deg,#141e30,#243b55);
padding:20px;
border-radius:15px;
text-align:center;
color:white;
font-size:20px;
box-shadow:0px 6px 15px rgba(0,0,0,0.4);
transition:0.3s;
}

.kpi-card:hover{
transform:scale(1.05);
}

</style>
""", unsafe_allow_html=True)

# ==========================
# LOAD DATA
# ==========================
data = pd.read_csv("dataset/ai4i2020.csv")

data = data.drop(["UDI", "Product ID"], axis=1)
data = pd.get_dummies(data, columns=["Type"])

X = data.drop("Machine failure", axis=1)
y = data["Machine failure"]

model = RandomForestClassifier()
model.fit(X, y)

# ==========================
# SIDEBAR INPUT
# ==========================
st.sidebar.header("⚙ Machine Input Parameters")

air_temp = st.sidebar.slider("Air Temperature (K)", 290.0, 310.0, 298.0)
process_temp = st.sidebar.slider("Process Temperature (K)", 300.0, 320.0, 308.0)
rpm = st.sidebar.slider("Rotational Speed (rpm)", 1000, 2000, 1500)
torque = st.sidebar.slider("Torque (Nm)", 20.0, 60.0, 40.0)
tool_wear = st.sidebar.slider("Tool Wear (min)", 0, 250, 10)

# ==========================
# DATASET PREVIEW
# ==========================
st.subheader("📊 Dataset Preview")
st.dataframe(data.head(), use_container_width=True)

# ==========================
# SENSOR TREND GRAPH
# ==========================
st.subheader("📈 Sensor Data Trend")

fig = px.line(
    data,
    y=["Air temperature [K]", "Process temperature [K]"],
    title="Temperature Trends"
)

st.plotly_chart(fig, use_container_width=True)

# ==========================
# PREDICTION
# ==========================
st.subheader("🔍 Fault Prediction")

if st.button("Check Machine Status"):

    input_data = pd.DataFrame({
        "Air temperature [K]": [air_temp],
        "Process temperature [K]": [process_temp],
        "Rotational speed [rpm]": [rpm],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear],
        "TWF": [0],
        "HDF": [0],
        "PWF": [0],
        "OSF": [0],
        "RNF": [0],
        "Type_H": [0],
        "Type_L": [1],
        "Type_M": [0]
    })

    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1]

    # ==========================
    # REMAINING USEFUL LIFE
    # ==========================
    st.subheader("⏳ Remaining Useful Life Prediction")

    MAX_MACHINE_LIFE = 1000
    machine_hours = st.slider("Machine Working Hours", 0, 1000, 500)

    remaining_life = MAX_MACHINE_LIFE - machine_hours

    st.metric("Remaining Machine Life", f"{remaining_life} Hours")

    if remaining_life < 100:
        st.error("🚨 Machine close to failure")
    elif remaining_life < 200:
        st.warning("⚠ Maintenance required soon")
    else:
        st.success("✅ Machine life is healthy")

    # ==========================
    # HEALTH SCORE
    # ==========================
    health_score = (1 - prob) * 100
    st.metric("🟢 Machine Health Score", f"{health_score:.2f}%")
    # ==========================
    # KPI DASHBOARD
    # ==========================
    st.subheader("📊 Machine KPI Dashboard")
    col1,col2,col3,col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
        🌡 Temperature<br>
        {air_temp:.2f} K
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
        ⚙ RPM<br>
        {rpm}
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
        💚 Health Score<br>
        {health_score:.2f}%
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
        ⚠ Failure Risk<br>
        {prob*100:.2f}%
        </div>
        """, unsafe_allow_html=True)


    # ==========================
    # FAILURE TYPE
    # ==========================
    failure_type = "No Failure"

    if tool_wear > 200:
        failure_type = "Tool Wear Failure (TWF)"
    elif process_temp - air_temp > 10:
        failure_type = "Heat Dissipation Failure (HDF)"
    elif torque > 55:
        failure_type = "Power Failure (PWF)"
    elif rpm < 1200 and torque > 50:
        failure_type = "Overstrain Failure (OSF)"

    if prediction[0] == 1:
        st.error(f"⚠ High Risk: {failure_type}")
    else:
        st.success("✅ Machine Operating Normally")
    #===========================
    #LIVE SENSOR GAUGE
    #===========================
    st.subheader("📡 Live Sensor Gauges")
    import plotly.graph_objects as go
    col1,col2 = st.columns(2)
    with col1:
        fig_temp = go.Figure(go.Indicator(
            mode="gauge+number",
            value=air_temp,
            title={'text': "Temperature"},
            gauge={
                'axis': {'range': [290,310]},
                'bar': {'color': "cyan"},
                'steps':[
                    {'range':[290,300],'color':"green"},
                    {'range':[300,305],'color':"yellow"},
                    {'range':[305,310],'color':"red"},
                ]
            }
    )   )

    st.plotly_chart(fig_temp,use_container_width=True)
    with col2:
        fig_rpm = go.Figure(go.Indicator(
            mode="gauge+number",
            value=rpm,
            title={'text': "Machine RPM"},
            gauge={
                'axis': {'range': [1000,2000]},
                'bar': {'color': "orange"},
                'steps':[
                    {'range':[1000,1400],'color':"green"},
                    {'range':[1400,1700],'color':"yellow"},
                    {'range':[1700,2000],'color':"red"},
                ]
            }
    )   )

    st.plotly_chart(fig_rpm,use_container_width=True)
    #===========================
    #AI RISK METER
    #===========================
    st.subheader("🤖 AI Failure Risk Meter")
    risk = prob * 100
    fig_risk = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk,
        title={'text': "Failure Probability (%)"},
        gauge={
            'axis': {'range':[0,100]},
            'bar': {'color':"red"},
            'steps':[
                {'range':[0,30],'color':"green"},
                {'range':[30,70],'color':"yellow"},
                {'range':[70,100],'color':"red"},
            ]
        }
    ))

    st.plotly_chart(fig_risk,use_container_width=True)
    # ==========================
    # MAINTENANCE RECOMMENDATION
    # ==========================
    st.subheader("🛠 Maintenance Recommendation")

    if tool_wear > 200:
        st.warning("🔧 Replace tool immediately")

    if torque > 55:
        st.warning("⚙ Reduce machine load")

    if process_temp > 315:
        st.warning("🌡 Cooling system check required")

    if prediction[0] == 0:
        st.info("✔ No immediate maintenance required")

# ==========================
# REAL TIME SENSOR SIMULATION
# ==========================
st.subheader("⏱ Real-Time Sensor Simulation")

live_temp = 298 + np.random.randn()
live_rpm = 1500 + np.random.randn() * 10

col1, col2 = st.columns(2)

col1.metric("Live Temperature", f"{live_temp:.2f} K")
col2.metric("Live RPM", f"{live_rpm:.0f}")

# ==========================
# VIBRATION MONITORING
# ==========================
st.subheader("📳 Vibration Monitoring")

vibration = st.slider("Machine Vibration Level", 0.0, 10.0, 2.5)

if vibration < 4:
    st.success("🟢 Normal vibration level")
elif vibration < 7:
    st.warning("🟡 High vibration detected")
else:
    st.error("🔴 Critical vibration! Possible mechanical fault")

# ==========================
# MULTI MACHINE DASHBOARD
# ==========================
st.subheader("🏭 Factory Machine Monitoring Dashboard")

num_machines = st.slider("Select Number of Machines", 3, 100, 10)

results = []

for i in range(num_machines):

    temp = 298 + np.random.randn()*2
    process_temp_sim = 308 + np.random.randn()*2
    rpm_sim = 1500 + np.random.randn()*50
    torque_sim = 40 + np.random.randn()*10
    tool_wear_sim = np.random.randint(0, 250)

    if np.random.rand() < 0.25:
        torque_sim = np.random.uniform(55, 70)
        tool_wear_sim = np.random.randint(200, 250)
        rpm_sim = np.random.uniform(900, 1100)

    input_data = pd.DataFrame({
        "Air temperature [K]": [temp],
        "Process temperature [K]": [process_temp_sim],
        "Rotational speed [rpm]": [rpm_sim],
        "Torque [Nm]": [torque_sim],
        "Tool wear [min]": [tool_wear_sim],
        "TWF": [0],
        "HDF": [0],
        "PWF": [0],
        "OSF": [0],
        "RNF": [0],
        "Type_H": [0],
        "Type_L": [1],
        "Type_M": [0]
    })

    prob = model.predict_proba(input_data)[0][1]
    health = (1 - prob) * 100

    if health > 80:
        status = "🟢 Healthy"
    elif health > 50:
        status = "🟡 Warning"
    else:
        status = "🔴 Critical"

    results.append([
        f"Machine {i+1}",
        round(temp,2),
        round(rpm_sim,0),
        round(torque_sim,2),
        f"{health:.2f}%",
        status
    ])

df_results = pd.DataFrame(results, columns=[
    "Machine",
    "Temperature",
    "RPM",
    "Torque",
    "Health Score",
    "Status"
])

st.dataframe(df_results, use_container_width=True)

# ==========================
# FACTORY HEALTH SUMMARY
# ==========================
st.subheader("📊 Factory Health Summary")

healthy_count = df_results["Status"].str.contains("Healthy").sum()
warning_count = df_results["Status"].str.contains("Warning").sum()
critical_count = df_results["Status"].str.contains("Critical").sum()

col1, col2, col3 = st.columns(3)

col1.metric("🟢 Healthy Machines", healthy_count)
col2.metric("🟡 Warning Machines", warning_count)
col3.metric("🔴 Critical Machines", critical_count)

# ==========================
# SORTED MACHINE RISK
# ==========================
st.subheader("📊 Machines Sorted by Risk")

df_sorted = df_results.copy()
df_sorted["Health Score"] = df_sorted["Health Score"].str.replace('%','').astype(float)
df_sorted = df_sorted.sort_values(by="Health Score")

st.dataframe(df_sorted, use_container_width=True)

# ==========================
# CRITICAL MACHINES
# ==========================
st.subheader("🚨 Top Critical Machines")

critical_machines = df_sorted[df_sorted["Health Score"] < 50]

if not critical_machines.empty:
    st.dataframe(critical_machines, use_container_width=True)
else:
    st.success("✅ No critical machines detected")

# ==========================
# DOWNLOAD REPORT
# ==========================
st.subheader("📄 Download Report")

csv = df_sorted.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download Machine Report",
    data=csv,
    file_name="machine_health_report.csv",
    mime="text/csv"
)
# ==========================
# FACTORY HEALTH SUMMARY
# ==========================
st.subheader("📊 Factory Health Summary")

healthy_count = df_results["Status"].str.contains("Healthy").sum()
warning_count = df_results["Status"].str.contains("Warning").sum()
critical_count = df_results["Status"].str.contains("Critical").sum()

col1, col2, col3 = st.columns(3)

col1.metric("🟢 Healthy Machines", healthy_count)
col2.metric("🟡 Warning Machines", warning_count)
col3.metric("🔴 Critical Machines", critical_count)
# ==========================
# MACHINE HEALTH GRAPH
# ==========================
st.subheader("📈 Machine Health Comparison")

fig_health = px.bar(
    df_sorted,
    x="Machine",
    y="Health Score",
    color="Health Score",
    title="Machine Health Score Distribution"
)

st.plotly_chart(fig_health, use_container_width=True)

# ==========================
# 3D MACHINE MODEL
# ==========================
st.subheader("🏭 3D Machine Model")

html_code = """
<script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>

<model-viewer 
src="machine.glb"
auto-rotate 
camera-controls 
style="width:100%; height:500px;">
</model-viewer>
"""

components.html(html_code, height=500)