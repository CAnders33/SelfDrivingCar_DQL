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
import csv
from shapely.geometry import Point, Polygon
import math

class RewardConfig:
    def __init__(self):
        # Balanced rewards from reached800 but scaled down to prevent overfitting - works with notmalization of calebs
        self.speed_reward = 0.03     # Weight for speed reward
        self.distance_reward = 1.5    # Weight for distance covered
        self.checkpoint_reward = 200  # Reduced from 300
        self.off_track_penalty = -100  # Less extreme penalty
        self.living_cost = -0.03     # Small cost to encourage progress
        self.brake_penalty = -0.5    # Reduced brake penalty
        self.idle_penalty = -0.8     # Penalty for very low speed

class CustomRenderer:
    def __init__(self, env, render_mode="rgb_array"):
        self.env = env
        self.render_mode = render_mode
        self.enabled = render_mode == "human"
        
        # Only initialize if in human mode
        if self.enabled:
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
        if not self.enabled:
            return (0, 0)
        screen_x = int(self.screen_width / 2 + world_x * self.zoom)
        screen_y = int(self.screen_height / 2 - world_y * self.zoom)
        return screen_x, screen_y

    def modify_frame(self, frame, car_position=None, car_angle=None, additional_info=None):
        if not self.enabled:
            return frame

        if frame is None or frame.size == 0:
            return np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

        try:
            modified = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            track_mask = gray > 150
            grass_mask = ~track_mask
            modified = np.zeros_like(frame)
            modified[track_mask] = self.track_color
            modified[grass_mask] = self.grass_color

            if self.overlay_enabled and car_position is not None:
                world_x, world_y = car_position
                screen_x, screen_y = self.world_to_screen(world_x, world_y)
                
                if 0 <= screen_x < self.screen_width and 0 <= screen_y < self.screen_height:
                    cv2.circle(modified, (screen_x, screen_y), self.car_radius, (255, 0, 0), -1)
                    if car_angle is not None:
                        angle_rad = np.radians(car_angle)
                        end_x = int(screen_x + self.car_radius * 2 * np.cos(angle_rad))
                        end_y = int(screen_y + self.car_radius * 2 * np.sin(angle_rad))
                        cv2.line(modified, (screen_x, screen_y), (end_x, end_y), (0, 255, 0), 1)

            if self.show_debug and additional_info:
                cv2.putText(modified, str(additional_info), (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            return modified
        except Exception as e:
            print(f"Error in modify_frame: {e}")
            return frame

    def render(self, frame, car_position=None, info=None):
        if not self.enabled:
            return

        try:
            if frame is None:
                return

            mod_frame = self.modify_frame(frame, car_position, info)
            window_name = "CarRacing Simulation"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, mod_frame)
            cv2.waitKey(1)
        except Exception as e:
            print(f"Error in render: {e}")

class carSim(gym.Wrapper):
    def __init__(self, seed=None, reward_config=None, render_mode="rgb_array", checkpoints=[]):
        env = gym.make("CarRacing-v3", render_mode=render_mode)
        super().__init__(env)
        
        self.renderer = CustomRenderer(env, render_mode)
        self._render_mode = render_mode  # Use different name to avoid property conflict
        self.reward_config = reward_config or RewardConfig()
        self.last_position = None
        self.show_rays = self._render_mode == "human"
        self.episode_distance = 0.0
        self.last_action = None
        self.check_points = checkpoints.copy()
        
        if seed is not None:
            self.seed_value = seed
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
            np.random.seed(seed)

    def reset(self, checkpoints=[], **kwargs):
        obs, info = super().reset(**kwargs)
        self.last_position = self.unwrapped.car.hull.position.copy()
        self.current_obs = obs
        self.episode_distance = 0.0
        self.check_points = checkpoints.copy()
        return obs, info

    def is_car_in_tile(self, car_position, tile):
        for fixture in tile.fixtures:
            shape = fixture.shape
            if shape.type == shape.e_polygon:
                poly_coords = [tuple(tile.transform * vertex) for vertex in shape.vertices]
                polygon = Polygon(poly_coords)
                return polygon.contains(Point(car_position))
        return False

    def step(self, action):
        obs, base_reward, done, truncated, info = self.env.step(action)
        self.current_obs = obs

        car = self.unwrapped.car
        current_position = car.hull.position
        
        # Check if car is on track
        wheel0_tiles = car.wheels[0].tiles
        wheel1_tiles = car.wheels[1].tiles
        wheel2_tiles = car.wheels[2].tiles
        wheel3_tiles = car.wheels[3].tiles
        left_side_on_road = (len(wheel0_tiles) > 0 and len(wheel2_tiles) > 0)
        right_side_on_road = (len(wheel1_tiles) > 0 and len(wheel3_tiles) > 0)
        on_road = left_side_on_road or right_side_on_road

        # Calculate rewards
        speed = np.linalg.norm(car.hull.linearVelocity)
        speed_reward = speed * self.reward_config.speed_reward

        # Distance reward
        if self.last_position is not None:
            curr_pos = np.array([current_position.x, current_position.y])
            last_pos = np.array([self.last_position.x, self.last_position.y])
            distance = np.linalg.norm(curr_pos - last_pos)
            distance_reward = distance * self.reward_config.distance_reward
            self.episode_distance += distance
        else:
            distance_reward = 0

        # Checkpoint reward
        checkpoint_reward = 0
        if len(self.check_points) > 0:
            next_tile_index = self.check_points[0]
            tile = self.unwrapped.road[next_tile_index]
            if self.is_car_in_tile(current_position, tile):
                self.check_points.pop(0)
                checkpoint_reward = self.reward_config.checkpoint_reward

        # Additional penalties
        brake_penalty = self.reward_config.brake_penalty if action[2] > 0.1 else 0
        idle_penalty = self.reward_config.idle_penalty if speed < 1.0 else 0
        
        # Action transition penalty (from reached800)
        action_transition_penalty = 0
        if self.last_action is not None:
            if (self.last_action[1] > 0.5 and action[2] > 0.1):  # was accelerating, now braking
                action_transition_penalty = -0.2

        self.last_action = action.copy()

        # Combine rewards
        reward = (speed_reward + 
                 distance_reward + 
                 checkpoint_reward + 
                 brake_penalty + 
                 idle_penalty + 
                 action_transition_penalty + 
                 self.reward_config.living_cost)

        # Off-track penalty and termination
        if not on_road:
            reward = self.reward_config.off_track_penalty
            done = True

        self.last_position = current_position.copy()
        return obs, reward, done, truncated, info

    def get_lidar_readings(self, frame):
        if frame is None or frame.size == 0:
            return {170: 1, 150: 1, 130: 1, 110: 1, 90: 1, 70: 1, 50: 1, 30: 1, 10: 1}

        h, w, _ = frame.shape
        car_y, car_x = int(h * 0.75), int(w * 0.5)
        directions = [170, 150, 130, 110, 90, 70, 50, 30, 10]
        distances = {}

        for angle in directions:
            if self.show_rays:
                distance = self.show_cast_ray(frame, car_x, car_y, angle)
            else:
                distance = self.cast_ray(frame, car_x, car_y, angle)
            distances[angle] = distance

        if self.show_rays:
            debug_frame = frame.copy()
            cv2.circle(debug_frame, (car_x, car_y), 2, (255, 0, 0), -1)
            for angle in directions:
                end_x = int(car_x + distances[angle] * np.cos(np.radians(angle)))
                end_y = int(car_y - distances[angle] * np.sin(np.radians(angle)))
                cv2.line(debug_frame, (car_x, car_y), (end_x, end_y), (255, 255, 0), 1)
                cv2.circle(debug_frame, (end_x, end_y), 3, (0, 0, 255), -1)
            cv2.namedWindow("LIDAR Debug", cv2.WINDOW_NORMAL)
            cv2.imshow("LIDAR Debug", debug_frame)
            cv2.waitKey(1)
            
        return distances

    def cast_ray(self, frame, start_x, start_y, angle, max_distance=100, min_distance=1):
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
        if not self.show_rays:
            return self.cast_ray(frame, start_x, start_y, angle, max_distance, min_distance)
            
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        frame_copy = frame.copy()

        cv2.circle(frame_copy, (start_x, start_y), 2, (0, 255, 0), -1)

        for d in range(min_distance, max_distance):
            x = int(start_x + d * cos_a)
            y = int(start_y - d * sin_a)

            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                pixel = frame[y, x]
                if not self.is_road(pixel):
                    cv2.line(frame_copy, (start_x, start_y), (x, y), (255, 255, 0), 1)
                    cv2.circle(frame_copy, (x, y), 3, (0, 0, 255), -1)
                    return d
        return max_distance

    def is_road(self, pixel):
        b, g, r = pixel
        if g > 100 and g > r + 30 and g > b + 30:
            return False
        return True

    def render(self):
        raw_frame = self.env.render()
        if raw_frame is None:
            return
        
        car = self.unwrapped.car
        if car and car.hull:
            car_pos = (car.hull.position.x, car.hull.position.y)
            car_angle = np.degrees(car.hull.angle)
            speed = np.linalg.norm(car.hull.linearVelocity)
            
            info = f"Speed: {speed:.2f} m/s | Angle: {car_angle:.1f}°"
            if self.show_rays:
                lidar = self.get_lidar_readings(self.current_obs)
                lidar_str = ", ".join(f"{int(v)}" for v in lidar.values())
                info += f" | LiDAR: [{lidar_str}]"
        else:
            car_pos = None
            car_angle = None
            info = "No car data"

        if self._render_mode == "human":
            self.renderer.render(raw_frame, car_pos, info)

class DQL:
    learning_rate = 0.0003  # Learning rate (from reached800)
    gamma = 0.99            # Discount factor
    epsilon = 1.0           # Start with full exploration
    epsilon_decay = 0.9971   # Decay rate
    buffer_size = 50000     # Replay buffer size
    batch_size = 128        # Batch size
    target_update = 1000    # Update target network frequency
    LIDAR_ANGLES = [170, 150, 130, 110, 90, 70, 50, 30, 10]
    DIRECTION_VECTORS = [(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in LIDAR_ANGLES]

    def __init__(self, state_dim, action_dim, env):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = []
        self.buffer = []
        self.steps = 0
        self.env = env
        self.eps = 1e-6
        
        # Initialize tensor cache
        self._state_tensor_cache = {}

        self.q_network = self.build_network()
        self.target_network = self.build_network()
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Keep target network in eval mode

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate, weight_decay=0.01)
        self.loss_fn = nn.HuberLoss(reduction='mean', delta=1.0)

        # Optimized action space from reached800
        STEER_SOFT = 0.30
        GAS_FULL = 1.0
        GAS_COAST = 0.5
        BRAKE_MODERATE = 0.8

        self.actions = [
            [-STEER_SOFT, GAS_FULL, 0.0],    # gentle left + gas
            [STEER_SOFT, GAS_FULL, 0.0],     # gentle right + gas
            [0.0, GAS_FULL, 0.0],            # straight + gas
            [-STEER_SOFT, GAS_COAST, 0.0],   # gentle left + coast
            [STEER_SOFT, GAS_COAST, 0.0],    # gentle right + coast
            [0.0, GAS_COAST, 0.0],           # straight + coast
            [0.0, 0.0, BRAKE_MODERATE],      # brake
        ]

    def build_network(self):
        model = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim)
        )
        return model

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)
        
        # Use cached tensor if available - training became super slow
        state_key = hash(state.tobytes())
        if state_key in self._state_tensor_cache:
            state_tensor = self._state_tensor_cache[state_key]
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            self._state_tensor_cache[state_key] = state_tensor

        # Switch to eval mode for inference
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        self.q_network.train()
        return q_values.argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        # Use cached tensors
        state_key = hash(state.tobytes())
        next_state_key = hash(next_state.tobytes())
        
        if state_key in self._state_tensor_cache:
            state_tensor = self._state_tensor_cache[state_key]
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            self._state_tensor_cache[state_key] = state_tensor

        if next_state_key in self._state_tensor_cache:
            next_state_tensor = self._state_tensor_cache[next_state_key]
        else:
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            self._state_tensor_cache[next_state_key] = next_state_tensor

        with torch.no_grad():
            
            self.q_network.eval() # Temporarily set to eval mode - batch norm errors bc only passing one sample
            current_q = self.q_network(state_tensor)[0][action]
            next_q = self.target_network(next_state_tensor).max()
            
            self.q_network.train() # Return to train mode
            
            target_q = reward + (1 - done) * self.gamma * next_q
            td_error = abs(target_q - current_q) + self.eps

        self.memory.append((state, action, reward, next_state, done, td_error.item()))
        if len(self.memory) > self.buffer_size:
            self.memory.pop(0)
            
        # Clean cache if too large
        if len(self._state_tensor_cache) > 1000:
            self._state_tensor_cache.clear()


    def train(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        # Calculate sampling probabilities
        priorities = np.array([exp[5] for exp in self.memory])
        probs = priorities / priorities.sum()

        # Sample batch using priorities
        batch_indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        batch = [self.memory[idx] for idx in batch_indices]
        
        states, actions, rewards, next_states, dones, _ = zip(*batch)

        # Convert to tensors efficiently
        states = torch.FloatTensor(np.array(states, dtype=np.float32))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states, dtype=np.float32))
        dones = torch.FloatTensor(dones)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_actions = self.q_network(next_states).argmax(1)
        max_next_q_values = self.target_network(next_states)\
                        .gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        loss = self.loss_fn(current_q_values, target_q_values.detach())
        loss_val = loss.item()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        return loss_val

    def train_agent(self, episodes, csv_path=None, checkpoints=[]):
        # Buffer for CSV writes
        csv_buffer = []
        csv_flush_frequency = 10

        for episode in range(episodes):
            state, _ = self.env.reset(checkpoints=checkpoints.copy())
            total_reward = 0
            done = False
            truncated = False
            start_time = time.time()

            # Get initial state vector
            lidar_readings = self.env.get_lidar_readings(state)
            car = self.env.unwrapped.car
            
            state_vector = np.array(list(lidar_readings.values()), dtype=np.float32) / 100.0
            while not (done or truncated):
                self.steps += 1
                action = self.select_action(state_vector)
                continuous_action = np.array(self.actions[action], dtype=np.float32)
                
                next_obs, reward, done, truncated, _ = self.env.step(continuous_action)
                next_lidar = self.env.get_lidar_readings(next_obs)
                next_state_vector = np.array(list(next_lidar.values()), dtype=np.float32) / 100.0
                
                self.store_experience(state_vector, action, reward, next_state_vector, done or truncated)
                last_loss = self.train()
                
                state = next_obs
                state_vector = next_state_vector
                total_reward += reward

            elapsed_time = time.time() - start_time
            distance = self.env.episode_distance
            avg_speed = distance / elapsed_time if elapsed_time > 0 else 0
            checkpoints_left = len(self.env.check_points)

            # Buffer CSV data
            if csv_path:
                csv_buffer.append([
                    episode + 1,        # Episode
                    total_reward,       # Reward
                    avg_speed,          # Speed
                    distance,           # Distance
                    checkpoints_left,   # Checkpoints Left
                    last_loss,          # Loss
                    elapsed_time,       # Time
                    self.epsilon        # Epsilon
                ])
                if len(csv_buffer) >= csv_flush_frequency:
                    with open(csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows(csv_buffer)
                    csv_buffer.clear()

            print(f"Episode {episode + 1:3d} | Reward: {total_reward:7.2f} | "
                  f"Speed: {avg_speed:5.2f} m/s | Dist: {distance:6.2f} m | "
                  f"CP Left: {checkpoints_left:2d} | Loss: {last_loss:.4f} | "
                  f"Eps: {self.epsilon:.4f}")
            
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

    def save_model(self, filename):
        os.makedirs('nets', exist_ok=True)
        torch.save(self.q_network.state_dict(), f'nets/{filename}.pth')
        print(f"Model saved to nets/{filename}.pth")

    def load_model(self, filename):
        self.q_network.load_state_dict(torch.load(f'nets/{filename}.pth'))
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

if __name__ == "__main__":
    reward_config = RewardConfig()
    SEED = 617
    checkpoints = [24, 54, 68, 100, 112, 118, 130, 145, 160, 180, 215, 243, 265]
    
    print("\nAvailable render modes:")
    print("1. rgb_array (with custom visualization)")
    print("2. human (native pygame window)")
    render_choice = input("Enter render mode number (or press Enter for rgb_array): ").strip()
    
    render_mode = "human" if render_choice == "2" else "rgb_array"
    
    env = carSim(seed=SEED, reward_config=reward_config, render_mode=render_mode, checkpoints=checkpoints.copy())
    observation, info = env.reset(checkpoints=checkpoints.copy())

    state_dim = 9  # Number of LIDAR beams
    action_dim = 7  # Number of discrete actions
    agent = DQL(state_dim, action_dim, env)

    # Model loading
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

    # CSV handling
    csv_path = None
    if not os.path.exists('data'):
        os.makedirs('data')
        
    data_files = [f for f in os.listdir('data') if f.endswith('.csv')]
    if data_files:
        print("\nPrior Data Save Files:")
        for i, file in enumerate(data_files):
            print(f"{i + 1}. {file}")

        choice = input("\nEnter file number to append to (or press Enter to create new file): ").strip()

        if choice and choice.isdigit() and 1 <= int(choice) <= len(data_files):
            csv_path = os.path.join('data', data_files[int(choice) - 1])
            print(f"Appending to existing file: {csv_path}")
        else:
            filename = input("Enter new filename (without .csv): ").strip()
            csv_path = os.path.join('data', f"{filename or 'episode_data'}.csv")
            print(f"New file will be created: {csv_path}")
    else:
        filename = input("Enter new filename (without .csv): ").strip()
        csv_path = os.path.join('data', f"{filename or 'episode_data'}.csv")

    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Episode", "Total Reward", "Avg Speed (m/s)", 
                "Distance (m)", "Checkpoints Left", "Loss",
                "Time (s)", "Epsilon"
            ])

    try:
        print("Starting training... Press Ctrl+C to stop")
        for i in range(500):
            # Reset epsilon at the start of each training cycle
            agent.epsilon = 1.0
            agent.train_agent(episodes=200, csv_path=csv_path, checkpoints=checkpoints.copy())
        
        save_name = input("\nEnter filename to save model (without .pth): ").strip() or 'car_dql_model'
        agent.save_model(save_name)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        save_name = input("\nEnter filename to save interrupted model: ").strip() or 'car_dql_model_interrupted'
        agent.save_model(save_name)
    
    finally:
        env.close()
