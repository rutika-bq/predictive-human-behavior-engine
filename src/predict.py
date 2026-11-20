import joblib
import pandas as pd

# Load the model
model = joblib.load('models/behavior_model.pkl')

# Create sample input
sample_input = pd.DataFrame({
    'click_speed': [250],
    'idle_time': [15],
    'typing_speed': [70],
    'page_switch_count': [6],
    'error_count': [3],
    'session_duration': [35],
    'task_completion_rate': [0.8]
})

# Make prediction
prediction = model.predict(sample_input)

# Print the results
targets = ['rage_quit', 'attention_loss', 'burnout_risk']
print("Prediction results:")
for i, target in enumerate(targets):
    print(f"{target}: {prediction[0][i]}")