import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
df = pd.read_csv('data/simulated_user_sessions.csv')

# Features
features = ['click_speed', 'idle_time', 'typing_speed', 'page_switch_count', 'error_count', 'session_duration', 'task_completion_rate']
X = df[features]

# Targets - convert to binary (threshold at 0.5)
targets = ['rage_quit', 'attention_loss', 'burnout_risk']
y = df[targets].apply(lambda x: (x > 0.5).astype(int))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = MultiOutputClassifier(RandomForestClassifier(random_state=42))
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print accuracy for each target
for i, target in enumerate(targets):
    acc = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
    print(f"Accuracy for {target}: {acc:.4f}")

# Save the model
joblib.dump(model, 'models/behavior_model.pkl')
print("Model saved to models/behavior_model.pkl")