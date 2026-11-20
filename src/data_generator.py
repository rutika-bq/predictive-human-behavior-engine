import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of rows
n = 5000

# Generate base columns
click_speed = np.random.normal(200, 50, n).clip(0, 500)  # clicks per minute, clipped
idle_time = np.random.exponential(10, n)  # minutes
typing_speed = np.random.normal(60, 20, n).clip(0, 150)  # words per minute
page_switch_count = np.random.poisson(5, n)
error_count = np.random.poisson(2, n)
session_duration = np.random.normal(30, 10, n).clip(0, 100)  # minutes
task_completion_rate = np.random.beta(2, 2, n)  # 0 to 1

# Generate dependent columns based on conditions
rage_quit = np.clip((error_count / 10 + idle_time / 100) / 2, 0, 1)
attention_loss = np.clip(1 - (typing_speed / 150), 0, 1)  # higher when typing_speed low
burnout_risk = np.clip((session_duration / 100) * (1 - task_completion_rate), 0, 1)

# Create DataFrame
df = pd.DataFrame({
    'click_speed': click_speed,
    'idle_time': idle_time,
    'typing_speed': typing_speed,
    'page_switch_count': page_switch_count,
    'error_count': error_count,
    'session_duration': session_duration,
    'task_completion_rate': task_completion_rate,
    'rage_quit': rage_quit,
    'attention_loss': attention_loss,
    'burnout_risk': burnout_risk
})

# Save to CSV
df.to_csv('data/simulated_user_sessions.csv', index=False)

print("Dataset generated and saved to data/simulated_user_sessions.csv")