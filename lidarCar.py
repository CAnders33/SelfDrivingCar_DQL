import gymnasium as gym
import numpy as np
import random
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class RewardConfig:
    def __init__(self):
        self.off_track_penalty = -150       # Penalty for going off track
        self.speed_reward_weight = 0.02      # Weight for speed reward
        self.distance_reward_weight = 1   # Weight for distance covered
        self.living_cost = -0.1             # Penalty per step to discourage inaction

class carSim(gym.Wrapper):
    def __init__(self, seed=None, show_lidar=True, reward_config=None):
        env = gym.make("CarRacing-v3", render_mode="human")
        super().__init__(env)
        
        self.show_lidar = show_lidar
        self.reward_config = RewardConfig()
        self.last_position = None
        
        if seed is not None:
            self.seed_value = seed
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
            np.random.seed(seed)

    def reset(self, **kwargs):
        if hasattr(self, 'seed_value'):
            kwargs['seed'] = self.seed_value
        obs, info = super().reset(**kwargs)
        self.last_position = self.unwrapped.car.hull.position
        self.current_obs = obs
        return obs, info

    def step(self, action):
        obs, base_reward, done, truncated, info = self.env.step(action)
        self.current_obs = obs

        # Get car info
        car = self.unwrapped.car
        current_position = car.hull.position

        # Check wheel positions
        wheel0_tiles = car.wheels[0].tiles  # Left front
        wheel1_tiles = car.wheels[1].tiles  # Right front
        wheel2_tiles = car.wheels[2].tiles  # Left rear
        wheel3_tiles = car.wheels[3].tiles  # Right rear

        # Check if both wheels on either side are off the road
        left_side_on_road = (len(wheel0_tiles) > 0 and len(wheel2_tiles) > 0)
        right_side_on_road = (len(wheel1_tiles) > 0 and len(wheel3_tiles) > 0)
        
        on_road = left_side_on_road or right_side_on_road

        # Calculate rewards

        # average speed reward
        speed = np.linalg.norm(car.hull.linearVelocity)  # Speed in m/s
        speed_reward = speed * self.reward_config.speed_reward_weight

        # Distance covered reward
        if self.last_position is not None:
            # Extract x,y coordinates from Box2D vectors
            curr_pos = np.array([current_position.x, current_position.y])
            last_pos = np.array([self.last_position.x, self.last_position.y])
            # Calculate Euclidean distance
            distance = np.linalg.norm(curr_pos - last_pos)
            distance_reward = distance * self.reward_config.distance_reward_weight
        else:
            distance_reward = 0
        
        living_cost = self.reward_config.living_cost
        reward = base_reward + speed_reward + distance_reward + living_cost

        if not on_road:
            reward = self.reward_config.off_track_penalty
            done = True

        self.last_position = current_position
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

    def __init__(self, state_dim, action_dim, env):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = []
        self.buffer = []
        self.steps = 0
        self.env = env

        self.q_network = self.build_network()
        self.target_network = self.build_network()

        # Initialize target network with Q-network parameters
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def build_network(self):
        model = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
        return model

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state)
            return q_values.argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.buffer_size:
            self.memory.pop(0)

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert lists to numpy arrays - got a flag from tensor
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        # Convert numpy arrays to tensors
        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions)
        rewards = torch.from_numpy(rewards)
        next_states = torch.from_numpy(next_states)
        dones = torch.from_numpy(dones)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        max_next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        loss = self.loss_fn(current_q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically update target network
        if self.steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def train_agent(self, episodes):
        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            truncated = False

            while not (done or truncated):
                self.steps += 1
                # Convert observation to state vector (LiDAR distances)
                lidar_readings = self.env.get_lidar_readings(state)
                state_vector = np.array(list(lidar_readings.values()))
                
                action = self.select_action(state_vector)
                # Convert discrete action to continuous space
                continuous_action = self._discrete_to_continuous(action)
                next_obs, reward, done, truncated, _ = self.env.step(continuous_action)
                next_state_vector = np.array(list(self.env.get_lidar_readings(next_obs).values()))
                
                self.store_experience(state_vector, action, reward, next_state_vector, done or truncated)
                self.train()
                state = next_obs
                total_reward += reward

            print(f"Episode {episode + 1}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.3f}")
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)  # Decay epsilon with minimum value

    def _discrete_to_continuous(self, action):
        # Convert discrete actions to continuous actions for CarRacing environment
        if action == 0:  return np.array([-1.0, 0, 0])  # Left
        elif action == 1: return np.array([1.0, 0, 0])  # Right
        elif action == 2: return np.array([0, 0, 0])    # Straight
        elif action == 3: return np.array([0, 1.0, 0])  # Accelerate
        else: return np.array([0, 0, 0.8])              # Brake

    def save_model(self, filename):
        os.makedirs('nets', exist_ok=True)  # Create nets directory if it doesn't exist
        print("\nSaving model weights sample:")
        for name, param in self.q_network.named_parameters():
            if 'weight' in name:
                print(f"{name} first 5 values: {param.data[:5]}")
                break
        
        torch.save(self.q_network.state_dict(), f'nets/{filename}.pth')
        print(f"Model saved to nets/{filename}.pth")

    def load_model(self, filename):
        print("\nBefore loading weights sample:")
        for name, param in self.q_network.named_parameters():
            if 'weight' in name:
                print(f"{name} first 5 values: {param.data[:5]}")
                break
        
        self.q_network.load_state_dict(torch.load(f'nets/{filename}.pth'))
        self.q_network.eval()
        
        print("\nAfter loading weights sample:")
        for name, param in self.q_network.named_parameters():
            if 'weight' in name:
                print(f"{name} first 5 values: {param.data[:5]}")
                break


# Main function
if __name__ == "__main__":
    # change rewards in class above
    reward_config = RewardConfig()

    SEED = 37843
    env = carSim(seed=SEED, show_lidar=True, reward_config=reward_config)
    observation, info = env.reset()

    # Get input dimensions from LiDAR readings (5 distances)
    state_dim = 5  # One value for each LiDAR beam
    action_dim = len(DQL.actions)  # Number of discrete actions

    # Initialize DQL agent and try to load existing model
    agent = DQL(state_dim, action_dim, env)
    model_path = 'nets/car_dql_model.pth'
    if os.path.exists(model_path):
        print(f"\nFound existing model at {model_path}")
        try:
            agent.load_model("car_dql_model")
            print("Successfully loaded existing model")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting with fresh model")

    try:
        # Training loop
        print("Starting training... Press Ctrl+C to stop")
        agent.train_agent(episodes=400)  # Number of episodes to train
        
        # Save the trained model
        agent.save_model("car_dql_model")
        print("Model saved successfully")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save model on interruption
        agent.save_model("car_dql_model_interrupted")
        print("Model saved")

    finally:
        env.close()
