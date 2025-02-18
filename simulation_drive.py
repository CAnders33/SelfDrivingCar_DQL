import gymnasium as gym
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("car_dql_model.h5")
print("Loaded trained DQL model.")

# Create the environment
env = gym.make("CarRacing-v3", render_mode="human", domain_randomize=False)  

# Function to simulate LIDAR
def lidar_scan(observation):
    h, w, _ = observation.shape  
    car_y, car_x = int(h * 0.6), int(w * 0.5)  

    scan_angles = [-45, -30, -15, 0, 15, 30, 45]
    scan_distances = []  

    LOWER_GRAY = np.array([100, 100, 100])  
    UPPER_GRAY = np.array([200, 200, 200])  

    for angle in scan_angles:
        for dist in range(1, 50):
            offset_x = int(np.cos(np.radians(angle)) * dist)
            offset_y = int(np.sin(np.radians(angle)) * dist)

            scan_x = np.clip(car_x + offset_x, 0, w - 1)
            scan_y = np.clip(car_y + offset_y, 0, h - 1)

            pixel = observation[scan_y, scan_x]

            if not (np.all(pixel >= LOWER_GRAY) and np.all(pixel <= UPPER_GRAY)):  
                scan_distances.append(dist)  
                break
        else:
            scan_distances.append(50)  

    return scan_distances  

# Initialize environment
observation, info = env.reset(seed=42)

# Store new experiences
new_data = []

for _ in range(500):  # Drive for 500 steps
    # Get LIDAR distances
    distances = np.array(lidar_scan(observation)).reshape(1, -1)

    # Predict action using trained model
    action = model.predict(distances, verbose=0)[0]
    action = np.clip(action, [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0])

    # Take action and get results
    next_observation, reward, terminated, truncated, _ = env.step(action)

    # Get next state (LIDAR after taking action)
    next_distances = lidar_scan(next_observation)

    # Save experience (convert NumPy arrays to lists)
    new_data.append([distances.flatten().tolist(), action.tolist(), reward, next_distances])

    # Reset environment if needed
    if terminated or truncated:
        observation, info = env.reset()
    else:
        observation = next_observation

# Convert to DataFrame and append to CSV
df = pd.DataFrame(new_data, columns=["State", "Action", "Reward", "Next State"])
df.to_csv("training_data.csv", mode="a", header=False, index=False)

print(f"🚀 {len(new_data)} new experiences saved to 'training_data.csv' for future training!")

# Close the environment
env.close()
