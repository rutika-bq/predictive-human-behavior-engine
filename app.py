import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Page Configuration
st.set_page_config(page_title="Predictive Human Behavior Engine", layout="wide")

# Custom CSS for modern cards
st.markdown("""
    <style>
    .kpi-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #FF6B6B;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .kpi-card.high-risk {
        border-left-color: #FF6B6B;
        background-color: #ffe0e0;
    }
    .kpi-card.low-risk {
        border-left-color: #51CF66;
        background-color: #e6ffe6;
    }
    .kpi-title {
        font-size: 14px;
        font-weight: bold;
        color: #555;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: bold;
        margin-top: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and data
model = joblib.load('models/behavior_model.pkl')
df = pd.read_csv('data/simulated_user_sessions.csv')

# Main Title
st.title("🤖 Predictive Human Behavior Engine")
st.markdown("Advanced AI/ML Dashboard for User Behavior Risk Analysis")

# Sidebar
st.sidebar.title("⚙️ Control Panel")
st.sidebar.header("Enter User Session Data")

click_speed = st.sidebar.number_input("Click Speed", min_value=0.0, value=200.0)
idle_time = st.sidebar.number_input("Idle Time (minutes)", min_value=0.0, value=10.0)
typing_speed = st.sidebar.number_input("Typing Speed (WPM)", min_value=0.0, value=60.0)
page_switch_count = st.sidebar.number_input("Page Switch Count", min_value=0, value=5)
error_count = st.sidebar.number_input("Error Count", min_value=0, value=2)
session_duration = st.sidebar.number_input("Session Duration (minutes)", min_value=0.0, value=30.0)
task_completion_rate = st.sidebar.number_input("Task Completion Rate", min_value=0.0, max_value=1.0, value=0.5)

# Predict button
if st.sidebar.button("🔮 Predict", use_container_width=True):
    input_data = pd.DataFrame({
        'click_speed': [click_speed],
        'idle_time': [idle_time],
        'typing_speed': [typing_speed],
        'page_switch_count': [page_switch_count],
        'error_count': [error_count],
        'session_duration': [session_duration],
        'task_completion_rate': [task_completion_rate]
    })
    
    prediction = model.predict(input_data)
    st.session_state.prediction = prediction[0]
    st.session_state.show_results = True
    st.session_state.input_data = input_data.iloc[0]

# Display prediction results if available
if 'show_results' in st.session_state and st.session_state.show_results:
    pred = st.session_state.prediction
    input_vals = st.session_state.input_data
    
    st.header("📊 Prediction Results")
    col1, col2, col3 = st.columns(3)
    
    # Rage Quit
    rage_quit_risk = "High Risk 🔴" if int(pred[0]) == 1 else "Low Risk ✅"
    rage_quit_class = "high-risk" if int(pred[0]) == 1 else "low-risk"
    with col1:
        st.markdown(f"""
        <div class="kpi-card {rage_quit_class}">
            <div class="kpi-title">Rage Quit Risk</div>
            <div class="kpi-value">{rage_quit_risk}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Attention Loss
    attention_risk = "High Risk 🔴" if int(pred[1]) == 1 else "Low Risk ✅"
    attention_class = "high-risk" if int(pred[1]) == 1 else "low-risk"
    with col2:
        st.markdown(f"""
        <div class="kpi-card {attention_class}">
            <div class="kpi-title">Attention Loss Risk</div>
            <div class="kpi-value">{attention_risk}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Burnout Risk
    burnout_risk = "High Risk 🔴" if int(pred[2]) == 1 else "Low Risk ✅"
    burnout_class = "high-risk" if int(pred[2]) == 1 else "low-risk"
    with col3:
        st.markdown(f"""
        <div class="kpi-card {burnout_class}">
            <div class="kpi-title">Burnout Risk</div>
            <div class="kpi-value">{burnout_risk}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk Explanation
    st.header("📝 Risk Explanation")
    
    if int(pred[0]) == 1:
        st.write(f"🔴 **Rage Quit - High Risk:** High error count ({input_vals['error_count']}) and/or idle time ({input_vals['idle_time']:.1f} min) detected. User likely to abandon.")
    else:
        st.write(f"✅ **Rage Quit - Low Risk:** User has manageable error rates ({input_vals['error_count']}) and idle time ({input_vals['idle_time']:.1f} min).")
    
    if int(pred[1]) == 1:
        st.write(f"🔴 **Attention Loss - High Risk:** Low typing speed ({input_vals['typing_speed']:.1f} WPM) indicates reduced focus and alertness.")
    else:
        st.write(f"✅ **Attention Loss - Low Risk:** Typing speed ({input_vals['typing_speed']:.1f} WPM) suggests good focus and alertness.")
    
    if int(pred[2]) == 1:
        st.write(f"🔴 **Burnout Risk - High Risk:** Long session ({input_vals['session_duration']:.1f} min) with low task completion ({input_vals['task_completion_rate']:.2f}) shows exhaustion.")
    else:
        st.write(f"✅ **Burnout Risk - Low Risk:** Session duration ({input_vals['session_duration']:.1f} min) and completion rate ({input_vals['task_completion_rate']:.2f}) are healthy.")

# Dataset Overview
st.header("📈 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Sessions", len(df))
with col2:
    st.metric("Average Idle Time", f"{df['idle_time'].mean():.2f} min")
with col3:
    st.metric("Average Typing Speed", f"{df['typing_speed'].mean():.2f} WPM")
with col4:
    st.metric("Average Session Duration", f"{df['session_duration'].mean():.2f} min")

# Behavior Analytics
st.header("📊 Behavior Analytics")
col1, col2, col3 = st.columns(3)

with col1:
    fig_idle = px.histogram(df, x='idle_time', nbins=30, title='Idle Time Distribution',
                            color_discrete_sequence=['#FF6B6B'])
    fig_idle.update_layout(height=400)
    st.plotly_chart(fig_idle, use_container_width=True)

with col2:
    fig_typing = px.histogram(df, x='typing_speed', nbins=30, title='Typing Speed Distribution',
                              color_discrete_sequence=['#4ECDC4'])
    fig_typing.update_layout(height=400)
    st.plotly_chart(fig_typing, use_container_width=True)

with col3:
    fig_error = px.histogram(df, x='error_count', nbins=20, title='Error Count Distribution',
                             color_discrete_sequence=['#FFE66D'])
    fig_error.update_layout(height=400)
    st.plotly_chart(fig_error, use_container_width=True)

# Feature Importance
st.header("🔍 Feature Importance")
st.write("How much each feature influences each prediction model:")

# Define feature names and target names
feature_names = ['click_speed', 'idle_time', 'typing_speed', 'page_switch_count', 'error_count', 'session_duration', 'task_completion_rate']
target_names = ['Rage Quit', 'Attention Loss', 'Burnout Risk']

# Create columns for feature importance charts
col1, col2, col3 = st.columns(3)

# Extract feature importance from each estimator in MultiOutputClassifier
for idx, (col, target_name) in enumerate(zip([col1, col2, col3], target_names)):
    # Get feature importance from the estimator
    importances = model.estimators_[idx].feature_importances_
    
    # Create DataFrame for Plotly
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    # Create horizontal bar chart
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                 title=f'{target_name} Feature Importance',
                 color_discrete_sequence=['#6C5CE7'])
    fig.update_layout(height=400, showlegend=False)
    
    with col:
        st.plotly_chart(fig, use_container_width=True)

# Sample Dataset Preview
st.header("📋 Sample Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)