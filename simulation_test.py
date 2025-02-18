# import gymnasium as gym
# import numpy as np
# #
# # the render mode is used to display an image so you can see what is happening, otherwise the simulation will run in the background
# # domain_randomize: Changes the physics and appearance of the track randomly
# env = gym.make("CarRacing-v3", render_mode="human", domain_randomize=False)  

# # observation is the image itsefl
# # info contains metadata (not always used but useful for debugging)
# # seed ensures the car appears in the same spot all the time
# observation, info = env.reset(seed=42)




# def is_on_road(pixel):
#     # """ Check if a given pixel represents the road (gray color). """
#     # return np.all(pixel < [200, 200, 200]) and np.all(pixel > [100, 100, 100])

#     """ Check if a pixel falls within the road gray color range. """
#     LOWER_GRAY = np.array([100, 100, 100])  # Dark gray
#     UPPER_GRAY = np.array([200, 200, 200])  # Light gray

#     return np.all(pixel >= LOWER_GRAY) and np.all(pixel <= UPPER_GRAY)


# for _ in range(100000): # run only 1000 times

#     h, w, _ = observation.shape  # Get image dimensions
#     y, x = int(h * 0.6), int(w * 0.5)  # Approximate car position in the image
#     pixel = observation[y, x]
#     on_road = is_on_road(pixel)
#     print('ON ROAD?', on_road)

#     # action = env.action_space.sample()  # this just picks a random action, for testing purposes (In a real RL algorithm, you would replace this random action with one chosen by your trained policy.)
#     # If we only wanted to implement steer and acceleration
#     steering = np.random.uniform(-1.0, 1.0)
#     acceleration = np.random.uniform(0.0, 1.0)
#     brake = np.random.uniform(0.0, 1.0)

#     action = np.array([steering, steering, 0])

#     # env.step(action) executes the selected action in the environment.
#     # observation: The next state (e.g., an image of the new position on the track).
#     # reward: A numerical value indicating how good or bad the action was.
#     # terminated: True if the episode has ended (e.g., the car crashes or completes the track).
#     # truncated: True if the episode is forcefully stopped (e.g., max steps reached).
#     # info: Additional information about the step.
#     observation, reward, terminated, truncated, info = env.step(action)
#     print(observation)

#     # check if the epside is over
#     if terminated or truncated:
#         observation, info = env.reset()


# # close the environment
# env.close()

import gymnasium as gym
import numpy as np
import pandas as pd

# Create the environment
env = gym.make("CarRacing-v3", render_mode="human", domain_randomize=False)  

# Number of episodes and step limit
num_episodes = 1  
max_steps_per_episode = 500 

# Data collection list
data = []

def lidar_scan(observation):
    """ Simulate LIDAR by checking distances to track edges in different directions. """
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

for episode in range(num_episodes):
    observation, info = env.reset(seed=episode)  
    total_reward = 0  

    print(f"\nStarting Episode {episode + 1}/{num_episodes}")

    for step in range(max_steps_per_episode):
        distances = lidar_scan(observation)  
        min_distance = min(distances)  

        # Adaptive Steering
        steering = 0
        if distances[3] < 10:  
            if distances[0] > distances[-1]:  
                steering = 0.5  
            else:
                steering = -0.5  
        elif distances[0] < distances[-1]:  
            steering = 0.3  
        elif distances[-1] < distances[0]:  
            steering = -0.3  

        # Adaptive Acceleration and Braking
        acceleration = 0.8 if min_distance > 15 else 0.3  
        brake = 0.2 if min_distance < 10 else 0  

        action = np.array([steering, acceleration, brake])

        # Print debug info every 50 steps
        if step % 50 == 0:
            print(f"Step {step}: Steering {steering}, Accel {acceleration}, Brake {brake}")

        next_observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        next_distances = lidar_scan(next_observation)

        data.append([distances, action.tolist(), reward, next_distances])

        if terminated or truncated:
            print(f"Episode {episode + 1} ended early at step {step}. Total reward: {total_reward:.2f}")
            break

        observation = next_observation

print("\nSimulation complete.")

df = pd.DataFrame(data, columns=["State", "Action", "Reward", "Next State"])
df.to_csv("training_data.csv", index=False)

# df = pd.DataFrame(data, columns=["State", "Action", "Reward", "Next State"])
# df.to_csv("training_data.csv", mode="a", header=not pd.read_csv("training_data.csv").empty, index=False)


print("Data collection complete. Saved as 'training_data.csv'.")

env.close()

