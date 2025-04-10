import gymnasium as gym
import numpy as np
import random
import os
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class RewardConfig:
    def __init__(self):
        self.off_track_penalty = -110        # Penalty for going off track - overfits if too low. Can go fast and crash
        self.speed_reward_weight = 0.02      # Weight for speed reward
        self.distance_reward_weight = 2    # Weight for distance covered
        self.living_cost = -0.03             # Penalty per step to discourage inaction

class CustomRenderer:
    def __init__(self, env):
        self.env = env
        self.overlay_enabled = True
        self.track_color = (255, 255, 255)  # White for road
        self.grass_color = (0, 0, 0)        # Black for non-road
        self.screen_width = 96   # Default CarRacing screen width
        self.screen_height = 96  # Default CarRacing screen height
        self.scale_factor = 4.0  # Scale factor for world to screen coordinates
        self.zoom = 4.0         # Zoom factor for visualization
        self.car_radius = 2     # Radius of car circle in pixels
        self.show_debug = True  # Show additional debug information

    def world_to_screen(self, world_x, world_y):
        """Convert world coordinates to screen coordinates"""
        # Apply zoom and center the coordinates
        screen_x = int(self.screen_width / 2 + world_x * self.zoom)
        screen_y = int(self.screen_height / 2 - world_y * self.zoom)  # Flip y-axis
        return screen_x, screen_y

    def screen_to_world(self, screen_x, screen_y):
        """Convert screen coordinates to world coordinates"""
        world_x = (screen_x - self.screen_width / 2) / self.zoom
        world_y = -(screen_y - self.screen_height / 2) / self.zoom  # Flip y-axis
        return world_x, world_y

    def modify_frame(self, frame, car_position=None, car_angle=None, additional_info=None):
        """
        Modify the frame before displaying it with car position and orientation.
        """
        if frame is None or frame.size == 0:
            return np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

        try:
            modified = frame.copy()

            # Convert to grayscale first
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Create binary mask for road (white) and non-road (black)
            track_mask = gray > 150
            grass_mask = ~track_mask

            # Create RGB image with white road and black background
            modified = np.zeros_like(frame)
            modified[track_mask] = self.track_color
            modified[grass_mask] = self.grass_color

            # Draw car position if available
            if self.overlay_enabled and car_position is not None:
                world_x, world_y = car_position
                screen_x, screen_y = self.world_to_screen(world_x, world_y)
                
                # Draw car circle
                if 0 <= screen_x < self.screen_width and 0 <= screen_y < self.screen_height:
                    # Draw car body
                    cv2.circle(modified, (screen_x, screen_y), self.car_radius, (255, 0, 0), -1)
                    
                    # Draw direction indicator if angle available
                    if car_angle is not None:
                        angle_rad = np.radians(car_angle)
                        end_x = int(screen_x + self.car_radius * 2 * np.cos(angle_rad))
                        end_y = int(screen_y + self.car_radius * 2 * np.sin(angle_rad))
                        cv2.line(modified, (screen_x, screen_y), (end_x, end_y), (0, 255, 0), 1)

            # Add debug info
            if self.show_debug and additional_info:
                cv2.putText(modified, str(additional_info), (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            return modified
        except Exception as e:
            print(f"Error in modify_frame: {e}")
            return frame  # Return original frame if modification fails

    def render(self, frame, car_position=None, info=None):
        try:
            if frame is None:
                return

            mod_frame = self.modify_frame(frame, car_position, info)
            
            # Ensure window name is consistent
            window_name = "CarRacing Simulation"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, mod_frame)
            cv2.waitKey(1)
        except Exception as e:
            print(f"Error in render: {e}")

class carSim(gym.Wrapper):
    def __init__(self, seed=None, reward_config=None, renderer=None, render_mode="rgb_array"):
        env = gym.make("CarRacing-v3", render_mode=render_mode)
        super().__init__(env)
        
        self.renderer = renderer if renderer else CustomRenderer(env)
        self._render_mode = render_mode  # Use private variable instead of property
        
        self.reward_config = RewardConfig()
        self.last_position = None
        self.show_rays = self._render_mode == "human"  # Only show rays in human mode
        
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
        speed = np.linalg.norm(car.hull.linearVelocity)  # Speed in m/s
        speed_reward = speed * self.reward_config.speed_reward_weight

        # Distance covered reward
        if self.last_position is not None:
            curr_pos = np.array([current_position.x, current_position.y])
            last_pos = np.array([self.last_position.x, self.last_position.y])
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
        - Straight left (175°)
        - 45° left of forward (120°)
        - Straight ahead (90°)
        - 45° right of forward (60°)
        - Straight right (5°)
        """
        if frame is None or frame.size == 0:
            return {175: 1, 120: 1, 90: 1, 60: 1, 5: 1}

        h, w, _ = frame.shape
        car_y, car_x = int(h * 0.72), int(w * 0.5)  # Approximate car position
        
        # Ordered from left to right
        directions = [175, 120, 90, 60, 5]
        distances = {}

        for angle in directions:
            if self.show_rays:
                distance = self.show_cast_ray(frame, car_x, car_y, angle)
            else:
                distance = self.cast_ray(frame, car_x, car_y, angle)
            distances[angle] = distance

        if self.show_rays:
            # Create debug visualization
            debug_frame = frame.copy()
            cv2.circle(debug_frame, (car_x, car_y), 2, (255, 0, 0), -1)  # Car position
            for angle in directions:
                end_x = int(car_x + distances[angle] * np.cos(np.radians(angle)))
                end_y = int(car_y - distances[angle] * np.sin(np.radians(angle)))
                cv2.line(debug_frame, (car_x, car_y), (end_x, end_y), (255, 255, 0), 1)
                cv2.circle(debug_frame, (end_x, end_y), 3, (0, 0, 255), -1)
            cv2.namedWindow("LIDAR Debug", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("LIDAR Debug", w * 4, h * 4)  # Make window 4x larger
            cv2.imshow("LIDAR Debug", debug_frame)
            cv2.waitKey(1)
            
        return distances

    def cast_ray(self, frame, start_x, start_y, angle, max_distance=100, min_distance=1):
        """Cast a ray and return distance to track edge."""
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        for d in range(min_distance, max_distance):
            x = int(start_x + d * cos_a)
            y = int(start_y - d * sin_a)

            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]: 
                pixel = frame[y, x]
                if not self.is_road(pixel):                    
                    return d
        return max_distance

    def show_cast_ray(self, frame, start_x, start_y, angle, max_distance=100, min_distance=1):
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        frame_copy = frame.copy()  # Create a copy for visualization

        # Draw a small circle at the sensor position
        cv2.circle(frame_copy, (start_x, start_y), 2, (0, 255, 0), -1)  # Green dot

        for d in range(min_distance, max_distance):
            x = int(start_x + d * cos_a)
            y = int(start_y - d * sin_a)

            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                pixel = frame[y, x]
                if not self.is_road(pixel):
                    # Draw the ray line from start to hit point
                    cv2.line(frame_copy, (start_x, start_y), (x, y), (255, 255, 0), 1)
                    cv2.circle(frame_copy, (x, y), 3, (0, 0, 255), -1)  # Red dot
                    return d
        return max_distance

    def is_road(self, pixel):
        """
        Check if a pixel is part of the road based on RGB values
        """
        b, g, r = pixel  # OpenCV uses BGR order
        # Define "green" if G is high AND clearly above R and B
        if g > 100 and g > r + 30 and g > b + 30:
            return False  # It's green
        return True

    def render(self):
        raw_frame = self.env.render()  # This returns an RGB array or None in human mode
        
        if raw_frame is None:
            print("Warning: Received None frame from environment")
            return
        
        car = self.unwrapped.car
        if car and car.hull:
            car_pos = (car.hull.position.x, car.hull.position.y)
            car_angle = np.degrees(car.hull.angle)  # Convert angle to degrees
            
            # Add speed info for debugging
            speed = np.linalg.norm(car.hull.linearVelocity)
            info = f"Speed: {speed:.2f} m/s | Angle: {car_angle:.1f}°"
            if self.show_rays:
                # Add LiDAR readings if rays are enabled
                lidar = self.get_lidar_readings(self.current_obs)
                lidar_str = ", ".join(f"{int(v)}" for v in lidar.values())
                info += f" | LiDAR: [{lidar_str}]"
        else:
            car_pos = None
            car_angle = None
            info = "No car data"

        # Only use custom renderer for rgb_array mode
        mod_frame = self.renderer.modify_frame(raw_frame, car_pos, car_angle, info)
        self.renderer.render(mod_frame, None, None)
    
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
    actions = ['Left', 'Right', 'Straight', 'Accelerate', 'Coast', 'Brake']

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
                lidar_str = ", ".join(f"{int(x):3d}" for x in list(lidar_readings.values()))
                print(
                    # \r\033[K
                    f"\rEpisode {episode + 1:3d} | LiDAR: [{lidar_str}] | {self.actions[action]:9s}\t | Step: {self.steps:5d}\t | Total Reward: {total_reward:7.2f} | Epsilon: {self.epsilon:6.3f}",
                    end='',
                    flush=True
                )
                
                # Convert discrete action to continuous space
                continuous_action = self._discrete_to_continuous(action)
                next_obs, reward, done, truncated, _ = self.env.step(continuous_action)
                next_state_vector = np.array(list(self.env.get_lidar_readings(next_obs).values()))
                
                self.store_experience(state_vector, action, reward, next_state_vector, done or truncated)
                self.train()
                state = next_obs
                total_reward += reward

            # print("\r\033[K", end='')
            print()
            # print(f"Episode {episode + 1:3d} | Total Reward: {total_reward:7.2f} | Epsilon: {self.epsilon:6.3f}")
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)  # Decay epsilon with minimum value

    def _discrete_to_continuous(self, action):
        # Convert discrete actions to continuous actions for CarRacing environment
        if action == 0:  return np.array([-1.0, 0, 0])  # Left
        elif action == 1: return np.array([1.0, 0, 0])  # Right
        elif action == 2: return np.array([0, 0, 0])    # Straight
        elif action == 3: return np.array([0, 1.0, 0])  # Accelerate
        elif action == 4: return np.array([0, 0.5, 0])  # Coast
        else: return np.array([0, 0, 0.8])              # Brake

    def save_model(self, filename):
        os.makedirs('nets', exist_ok=True)  # Create nets directory if it doesn't exist
        print("\nSaving model weights sample:")  # chaeck this and output when loaded to see if it is the same
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

    SEED = np.random.randint(100) # 37843
    
    # Ask user for render mode
    print("\nAvailable render modes:")
    print("1. rgb_array (with custom visualization)")
    print("2. human (native pygame window)")
    render_choice = input("Enter render mode number (or press Enter for rgb_array): ").strip()
    
    render_mode = "human" if render_choice == "2" else "rgb_array"
    
    # Create environment with selected render mode
    env = carSim(seed=SEED, reward_config=reward_config, render_mode=render_mode)
    observation, info = env.reset()

    # Get input dimensions from LiDAR readings (5 distances)
    state_dim = 5  # One value for each LiDAR beam
    action_dim = len(DQL.actions)  # Number of discrete actions

    # Initialize DQL agent
    agent = DQL(state_dim, action_dim, env)

    # List available models and let user choose
    if os.path.exists('nets'):
        model_files = [f for f in os.listdir('nets') if f.endswith('.pth')]
        if model_files:
            print("\nAvailable models:")
            for i, model_file in enumerate(model_files):
                print(f"{i+1}. {model_file}")
            
            choice = input("\nEnter model number to load (or press Enter for new model): ")
            if choice.strip() and choice.isdigit() and 1 <= int(choice) <= len(model_files):
                model_name = model_files[int(choice)-1].replace('.pth', '')
                try:
                    agent.load_model(model_name)
                    print(f"Successfully loaded model: {model_name}")
                except Exception as e:
                    print(f"Error loading model: {e}")
                    print("Starting with fresh model")
            else:
                print("Starting with fresh model")
        else:
            print("\nNo models directory found. Starting with fresh model")
    else:
        print("\nNo models directory found. Starting with fresh model")

    try:
        # Training loop
        print("Starting training... Press Ctrl+C to stop")
        for i in range(3):
            agent.train_agent(episodes=200)  # Number of episodes to train
        
        # Get filename for saving model
        save_name = input("\nEnter filename to save model (without .pth, or press Enter for 'car_dql_model'): ")
        if not save_name.strip():
            save_name = "car_dql_model"
        
        # Save the trained model
        agent.save_model(save_name)
        print("Model saved successfully")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save model on interruption
        save_name = input("\nEnter filename to save interrupted model (without .pth, or press Enter for 'car_dql_model_interrupted'): ")
        if not save_name.strip():
            save_name = "car_dql_model_interrupted"
        agent.save_model(save_name)
        print("Model saved")

    finally:
        env.close()
