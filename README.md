## Live Demo

[Click here to view the app](https://predictive-human-behavior-engine-grrgxnpcemxctwqzweemjg.streamlit.app/)
# 🤖 Predictive Human Behavior Engine

An advanced AI/ML dashboard that analyzes user interaction data to predict behavioral risks including rage quit, attention loss, and burnout risk. Built with Python, scikit-learn, and Streamlit.

---

## 📋 Project Overview

The **Predictive Human Behavior Engine** is a comprehensive machine learning solution designed to understand and predict user behavior patterns. By analyzing key behavioral metrics such as typing speed, idle time, error rates, and session duration, the engine provides real-time risk assessments to help organizations identify and support users before they disengage or experience burnout.

This project demonstrates end-to-end ML workflow: data generation, model training, model deployment, and interactive visualization through a professional AI/ML dashboard.

---

## ✨ Features

- **🔮 Real-Time Predictions**: Input user behavior metrics and get instant risk assessments
- **📊 Interactive Dashboard**: Professional Streamlit UI with Plotly visualizations
- **🎯 Three Risk Predictions**:
  - **Rage Quit Risk**: Predicts likelihood of user abandonment
  - **Attention Loss Risk**: Detects reduced focus and alertness
  - **Burnout Risk**: Identifies user exhaustion patterns
- **📈 Behavior Analytics**: Visualize distributions of key behavioral metrics
- **🔍 Feature Importance**: Understand which factors drive each prediction
- **📋 Dataset Overview**: Quick metrics on simulated user sessions
- **📌 Sample Dataset Preview**: Explore real data used in training
- **💾 Pre-trained Models**: RandomForest-based MultiOutput classifier for multi-target prediction

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.x |
| **Data Processing** | pandas, NumPy |
| **Machine Learning** | scikit-learn |
| **Visualization** | Plotly, Streamlit |
| **Model Serialization** | joblib |

---

## 📁 Project Structure

```
predictive-human-behaviour-engine/
│
├── data/
│   └── simulated_user_sessions.csv       # Generated dataset (5000 rows)
│
├── models/
│   └── behavior_model.pkl                # Trained MultiOutputClassifier
│
├── src/
│   ├── data_generator.py                 # Generate synthetic user data
│   ├── train_model.py                    # Train ML models
│   └── predict.py                        # Standalone prediction script
│
├── app.py                                # Main Streamlit dashboard
├── README.md                             # This file
└── requirements.txt                      # Python dependencies
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/predictive-human-behaviour-engine.git
cd predictive-human-behaviour-engine
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pandas numpy scikit-learn streamlit plotly joblib
```

### Running the Project

**Option 1: Run the Interactive Dashboard**
```bash
streamlit run app.py
```
The dashboard will open in your browser at `http://localhost:8501`

**Option 2: Generate Fresh Data**
```bash
python src/data_generator.py
```
This creates a new `data/simulated_user_sessions.csv` with 5000 random user sessions.

**Option 3: Train a New Model**
```bash
python src/train_model.py
```
Trains the RandomForest MultiOutputClassifier and saves it to `models/behavior_model.pkl`

**Option 4: Make Predictions**
```bash
python src/predict.py
```
Tests the model with sample input data.

---

## 📊 Dashboard Features

### Prediction Results
- Input 7 behavioral metrics through the Control Panel sidebar
- Get instant predictions for three risk types
- Visual indicators: Low Risk ✅ or High Risk 🔴
- Contextual explanations based on input values

### Dataset Overview
- Total sessions count
- Average idle time
- Average typing speed
- Average session duration

### Behavior Analytics
- **Idle Time Distribution**: Histogram of idle time patterns
- **Typing Speed Distribution**: Histogram of typing speed metrics
- **Error Count Distribution**: Histogram of user error frequencies

### Feature Importance
- Visual ranking of feature importance for each model
- Understand which metrics most influence each prediction
- Separate charts for Rage Quit, Attention Loss, and Burnout Risk

### Sample Dataset Preview
- Browse the first 10 rows of actual training data

---

## 📈 Data Features

The model uses **7 input features** to make predictions:

| Feature | Description | Type |
|---------|-------------|------|
| `click_speed` | Clicks per minute | Continuous |
| `idle_time` | Minutes of inactivity | Continuous |
| `typing_speed` | Words per minute | Continuous |
| `page_switch_count` | Number of page switches | Integer |
| `error_count` | Number of errors made | Integer |
| `session_duration` | Total session time in minutes | Continuous |
| `task_completion_rate` | Percentage of tasks completed | Continuous (0-1) |

### Target Variables (Predicted)

| Target | Description |
|--------|-------------|
| `rage_quit` | 1 = High risk of abandonment, 0 = Low risk |
| `attention_loss` | 1 = High attention loss risk, 0 = Low risk |
| `burnout_risk` | 1 = High burnout risk, 0 = Low risk |

---

## 🤖 Model Architecture

- **Algorithm**: RandomForestClassifier (10 trees, random_state=42)
- **Multi-Output Strategy**: MultiOutputClassifier for parallel multi-target prediction
- **Data Split**: 80% training, 20% testing
- **Performance**: Model accuracy displayed for each target during training

---

## 🔄 Prediction Logic

The model uses conditional logic during data generation to ensure realistic relationships:

- **Rage Quit**: Increases with high error count and idle time
- **Attention Loss**: Increases when typing speed is low
- **Burnout Risk**: Increases with long sessions and low task completion rate

---

## 📸 Screenshots Section

### Dashboard Overview
The dashboard features three main sections:

1. **Prediction Results** - KPI cards showing risk levels
2. **Risk Explanation** - Context-specific insights
3. **Behavior Analytics** - Interactive data visualizations

---

## 🚀 Future Improvements

- [ ] Add user authentication and session management
- [ ] Implement model version control and A/B testing
- [ ] Create API endpoints for external integrations
- [ ] Add real user data integration
- [ ] Develop mobile-responsive UI
- [ ] Implement model retraining pipeline
- [ ] Add anomaly detection for unusual behavior
- [ ] Create automated alert system for high-risk users
- [ ] Deploy to cloud platforms (AWS, GCP, Azure)
- [ ] Add advanced feature engineering and selection
- [ ] Implement SHAP values for explainability
- [ ] Create comparative analysis across user cohorts

---

## 📝 Usage Examples

### Example 1: Normal User Session
```
Input:
- Click Speed: 200
- Idle Time: 10 min
- Typing Speed: 70 WPM
- Page Switches: 5
- Error Count: 2
- Session Duration: 30 min
- Task Completion: 0.8

Output:
- Rage Quit: Low Risk ✅
- Attention Loss: Low Risk ✅
- Burnout Risk: Low Risk ✅
```

### Example 2: At-Risk User
```
Input:
- Click Speed: 100
- Idle Time: 30 min
- Typing Speed: 20 WPM
- Page Switches: 15
- Error Count: 10
- Session Duration: 60 min
- Task Completion: 0.2

Output:
- Rage Quit: High Risk 🔴
- Attention Loss: High Risk 🔴
- Burnout Risk: High Risk 🔴
```

---

## 🛡️ Data Privacy

This project uses **synthetically generated data** for demonstration purposes. No real user data is included. When deployed with real data, ensure compliance with:

- GDPR (General Data Protection Regulation)
- CCPA (California Consumer Privacy Act)
- Local data protection regulations

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---


---

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for interactive dashboards
- Powered by [scikit-learn](https://scikit-learn.org/) for ML models
- Visualizations created with [Plotly](https://plotly.com/)

---

## 📞 Support

If you have questions or encounter issues:

1. Check the [Issues](https://github.com/yourusername/predictive-human-behaviour-engine/issues) page
2. Create a new issue with detailed information
3. Include error messages and steps to reproduce

---

## 🎯 Quick Links

- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn Guide](https://scikit-learn.org/stable/documentation.html)
- [Plotly Documentation](https://plotly.com/python/)

---

**Made with ❤️ for AI/ML enthusiasts**
