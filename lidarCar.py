import gymnasium as gym
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

class carSim(gym.Wrapper):
    def __init__(self, seed=None, show_lidar=True):
        env = gym.make("CarRacing-v3", render_mode="human")
        super().__init__(env)
        
        self.show_lidar = show_lidar
        
        if seed is not None:
            self.seed_value = seed
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
            np.random.seed(seed)

    def reset(self, **kwargs):
        if hasattr(self, 'seed_value'):
            kwargs['seed'] = self.seed_value
        obs, info = self.env.reset(**kwargs)
        self.current_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.current_obs = obs
        return obs, reward, done, truncated, info

    def get_lidar_readings(self, frame):
        """
        Simulated LiDAR using image processing to detect track edges.
        Returns distances for five beams:
        - Straight left (90°)
        - 45° left of forward (45°)
        - Straight ahead (0°)
        - 45° right of forward (-45°)
        - Straight right (-90°)
        """
        h, w, _ = frame.shape
        car_y, car_x = int(h * 0.6), int(w * 0.5)  # Approximate car position
        
        # Ordered from left to right
        directions = [90, 45, 0, -45, -90]  # Degrees relative to car
        distances = {}

        for angle in directions:
            distances[angle] = self.cast_ray(frame, car_x, car_y, angle)

        return distances

    def cast_ray(self, frame, start_x, start_y, angle, max_distance=100):
        """
        Cast a ray from the car in the given direction and return the distance to the track edge.
        """
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        for d in range(1, max_distance):  # Max check distance in pixels
            x = int(start_x + d * cos_a)
            y = int(start_y - d * sin_a)  # Invert y since images are top-down

            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]: 
                pixel = frame[y, x]
                if not self.is_road(pixel):  # Found track edge
                    return d
        return max_distance  # Default if no edge found

    def is_road(self, pixel):
        """
        Check if a pixel is part of the road based on RGB values
        """
        # Dark/grey color detection with higher tolerance
        return (70 < np.mean(pixel) < 140 and  # Average intensity
                np.std(pixel) < 30 and  # Color similarity
                pixel[1] < 150)  # Not too green

    def render(self):
        # Get the base frame from the environment
        frame = self.env.render()
        
        if self.show_lidar and hasattr(self, 'current_obs'):
            # Use the stored observation for LiDAR calculations
            distances = self.get_lidar_readings(self.current_obs)

            # Draw LiDAR rays
            x1, y1 = int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.6)  # Car position
            for angle, dist in distances.items():
                angle_rad = np.radians(angle)
                x2, y2 = int(x1 + dist * np.cos(angle_rad)), int(y1 - dist * np.sin(angle_rad))
                # Draw bright red rays for better visibility
                cv2.line(frame, (x1, y1), (x2, y2), (255, 50, 50), 2)

            # Draw car position indicator
            cv2.circle(frame, (x1, y1), 3, (50, 50, 255), -1)

        return frame


    def toggle_lidar(self):
        """ Toggle the LiDAR visualization on/off """
        self.show_lidar = not self.show_lidar
        return self.show_lidar

    def close(self):
        self.env.close()


class DQL:
    learning_rate = 0.001   # Learning rate for the neural network, alpha
    gamma = 0.9             # Discount factor, gamma
    epsilon = 0.1           # Epsilon greedy parameter
    epsilon_decay = 0.995   # Epsilon decay rate
    buffer_size = 10000     # Replay buffer size
    batch_size = 32         # Number of samples to take from memory
    target_update = 10      # Update target network after agent takes n steps

    # Actions taken every 3 frames by default with carracing-v3
    actions = ['Left', 'Right', 'Straight', 'Accelerate', 'Brake']

    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = []
        self.buffer = []

        self.q_network = self.build_network()
        self.target_network = self.build_network()

        # self.target_network.load_state_dict(self.q_network.state_dict())
        # self.target_network.eval()

    # Choose action using epsilon-greedy policy
    def select_action(self, state):
        if np.random.rand() < self.epsilon: # return random action  --  Explore
            return np.random.choice(self.action_dim)

        else: # return action with highest Q-value from Q-network  --  Exploit
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state)
            return q_values.argmax().item()

#     # Store experience in replay memory
#     function store_experience(state, action, reward, next_state, done):
#         Add (state, action, reward, next_state, done) to replay memory

#     # Train the Q-network using experience replay
#     function train():
#         if replay memory size < batch size:
#             return  # Do not train until enough data is collected

#         Sample a batch from replay memory
#         Compute target Q-values:
#             If done:
#                 target Q = reward
#             Else:
#                 target Q = reward + gamma * max(Q(next_state, all_actions)) from target network

#         Compute loss between predicted Q-values and target Q-values
#         Backpropagate loss and update Q-network

#         # Periodically update target network
#         if step % update_frequency == 0:
#             Copy weights from Q-network to target Q-network

#     # Main training loop
#     function train_agent(episodes):
#         for episode in range(episodes):
#             state = reset environment
#             done = False

#             while not done:
#                 action = select_action(state)
#                 next_state, reward, done = take action in environment
#                 store_experience(state, action, reward, next_state, done)
#                 train()  # Train the model
#                 state = next_state

#             Decay epsilon (reduce exploration over time)

#     # Save the trained model
#     function save_model(filename):
#         Save Q-network weights to file

#     # Load a trained model
#     function load_model(filename):
#         Load Q-network weights from file


# Main function
if __name__ == "__main__":
    import pygame
    
    SEED = 37843
    env = carSim(seed=SEED, show_lidar=True)  # Start with LiDAR visualization enabled
    observation, info = env.reset()

    print("Press 'L' to toggle LiDAR visualization, 'Q' to quit")
    running = True
    
    while running:
        # Handle input
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_l:  # L pressed
                    is_on = env.toggle_lidar()
                    print(f"LiDAR visualization {'enabled' if is_on else 'disabled'}")
                elif event.key == pygame.K_q:  # Q pressed
                    running = False
        
        if not running:
            break
            
        action = np.array([np.random.uniform(-1, 1), np.random.uniform(0, 1), 0])
        observation, reward, terminated, truncated, info = env.step(action)
        # print(action, "\t\t", reward)
        
        #print lidar readings
        lidar_readings = env.get_lidar_readings(observation)
        print("LiDAR Distances:", lidar_readings)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()
