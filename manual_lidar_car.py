import gymnasium as gym
import numpy as np
import cv2
import pygame
import time
import csv
import os
from shapely.geometry import Point, Polygon


class ManualCarSim(gym.Wrapper):
    def __init__(self, seed=None, show_lidar=True, checkpoints = []):
        # env = gym.make("CarRacing-v3", render_mode="human")
        env = gym.make("CarRacing-v3", render_mode="rgb_array")
        super().__init__(env)
        
        self.show_lidar = show_lidar
        self.last_position = None
        self.episode_distance = 0.0

        self.check_points = checkpoints


        
        if seed is not None:
            self.seed_value = seed
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
            np.random.seed(seed)

    def reset(self, checkpoints=[], **kwargs):
        if hasattr(self, 'seed_value'):
            kwargs['seed'] = self.seed_value
        obs, info = self.env.reset(**kwargs)
        self.current_obs = obs
        self.last_position = self.unwrapped.car.hull.position.copy()
        self.episode_distance = 0.0
        self.check_points = checkpoints

        return obs, info
    

    def is_car_in_tile(self, car_position, tile):
        for fixture in tile.fixtures:
            shape = fixture.shape
            # Check if the shape is a polygon (Box2D polygons have type e_polygon)
            if shape.type == shape.e_polygon:
                # Transform each vertex from local to world coordinates using the body's transform
                poly_coords = [tuple(tile.transform * vertex) for vertex in shape.vertices]
        polygon = Polygon(poly_coords)
        return polygon.contains(Point(car_position))
    

    def print_current_tile_index(self):
        """
        Returns the index of the road tile in which the car's center is located.
        If not found, returns None.
        """
        car_position = self.unwrapped.car.hull.position  # World coordinates (b2Vec2)
        # Make sure you're iterating over the correct list of tile bodies.
        # This example assumes self.env.unwrapped.road is the list containing your tile bodies.
        for idx, tile in enumerate(self.env.unwrapped.road):
            if self.is_car_in_tile(car_position, tile):
                print('Index tile', idx)


    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.current_obs = obs

        # Distance tracking
        car = self.unwrapped.car
        current_position = car.hull.position

        if self.last_position is not None:
            curr_pos = np.array([current_position.x, current_position.y])
            last_pos = np.array([self.last_position.x, self.last_position.y])
            distance = np.linalg.norm(curr_pos - last_pos)
            self.episode_distance += distance
        self.last_position = current_position.copy()

        # print current tile
        self.print_current_tile_index()

        if len(self.check_points) > 0:
            next_tile_index = self.check_points[0]
            tile = self.unwrapped.road[next_tile_index]
            if self.is_car_in_tile(current_position, tile):
                self.check_points.pop(0)
                print('in checkpoint!')


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
        car_y, car_x = int(h * 0.69), int(w * 0.5)  # Approximate car position
        
        # Ordered from left to right
        # directions = [180, 135, 90, 45, 0]
        directions = [175, 120, 90, 60, 5]
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

    
    def render(self):
        # Get the base frame from the environment
        frame = self.env.render()

        scale_x = 600/96
        scale_y = 400/96
        
        if self.show_lidar and hasattr(self, 'current_obs'):
            # Use the stored observation for LiDAR calculations
            distances = self.get_lidar_readings(self.current_obs)

            # Draw LiDAR rays
            x1, y1 = int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.69)  # Car position
            for angle, dist in distances.items():
                angle_rad = np.radians(angle)
                x2, y2 = int(x1 + dist * scale_x * np.cos(angle_rad)), int(y1 - dist * scale_y * np.sin(angle_rad))
                # Draw bright red rays for better visibility
                cv2.line(frame, (x1, y1), (int(x2), int(y2)), (255, 50, 50), 2)

            # Draw car position indicator
            cv2.circle(frame, (x1, y1), 3, (50, 50, 255), -1)

        return frame


    def is_road(self, pixel):
        """
        Check if a pixel is part of the road based on RGB values
        """
        # Dark/grey color detection with higher tolerance
        # return (70 < np.mean(pixel) < 140 and  # Average intensity
        #         np.std(pixel) < 30 and  # Color similarity
        #         pixel[1] < 150)  # Not too green

        b, g, r = pixel  # OpenCV uses BGR order
        # Define "green" if G is high AND clearly above R and B
        if g > 100 and g > r + 30 and g > b + 30:
            return False  # It's green
        return True

    def toggle_lidar(self):
        """ Toggle the LiDAR visualization on/off """
        self.show_lidar = not self.show_lidar
        return self.show_lidar
    

    def close(self):
        self.env.close()

def get_keyboard_action():
    """
    Get car control inputs from keyboard:
    - Arrow keys for steering and acceleration
    - Space for braking
    Returns: [steering, gas, brake]
    """
    keys = pygame.key.get_pressed()
    
    # Steering: left/right arrows (-1 is left, +1 is right)
    steering = 0.0
    if keys[pygame.K_LEFT]:
        steering = -1.0
    elif keys[pygame.K_RIGHT]:
        steering = 1.0
        
    # Gas: up arrow
    gas = 1.0 if keys[pygame.K_UP] else 0.0
    
    # Brake: space bar or down arrow
    brake = 1.0 if keys[pygame.K_SPACE] or keys[pygame.K_DOWN] else 0.0
    
    return np.array([steering, gas, brake])

# Main function
if __name__ == "__main__":
    import pygame
    pygame.init()
    pygame.display.set_mode((100, 100))
    
    SEED = 617 # 37843
    checkpoints = [7, 25, 30]
    env = ManualCarSim(seed=SEED, show_lidar=True, checkpoints=checkpoints.copy())  # Start with LiDAR visualization enabled
    observation, info = env.reset(checkpoints=checkpoints.copy())
    
    episode_total_reward = 0.0
    episode_start_time = time.time()

    print("Controls:")
    print("- Arrow keys: Steering (left/right) and Gas (up)")
    print("- Space or Down Arrow: Brake")
    print("- L: Toggle LiDAR visualization")
    print("- Q: Quit")
    
    if not os.path.exists('data'):
        os.makedirs('data')
    csv_path = os.path.join('data', 'manual_episode_data.csv')
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Distance (m)", "Time (s)", "Avg Speed (m/s)", "Total Reward"])
    episode_counter = 1

    
    running = True
    clock = pygame.time.Clock()
    frame_count = 0  # Add frame counter
    
    while running:
        # Handle input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_l:  # L pressed
                    is_on = env.toggle_lidar()
                    print(f"LiDAR visualization {'enabled' if is_on else 'disabled'}")
                elif event.key == pygame.K_q:  # Q pressed
                    running = False
        
        if not running:
            break
            
        # Get action from keyboard
        action = get_keyboard_action()
        
        # Step environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        episode_total_reward += reward

        frame = env.render()
        cv2.imshow("CarRacing with LiDARRR", frame)
        # cv2.putText(frame, "Custom LiDAR Frame", (10, 20),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        # cv2.waitKey(1) 

        
        # Get lidar readings and road status every 3 frames
        if frame_count % 3 == 0:
            lidar_readings = env.get_lidar_readings(observation)
            
            # Get road status
            car_env = env.unwrapped
            car = car_env.car

            # Check each wheel's tiles set
            wheel0_tiles = car.wheels[0].tiles
            wheel1_tiles = car.wheels[1].tiles
            wheel2_tiles = car.wheels[2].tiles
            wheel3_tiles = car.wheels[3].tiles

            # Complex two wheels on road
            left_side_on_road = (len(wheel0_tiles) > 0 and len(wheel2_tiles) > 0)
            right_side_on_road = (len(wheel1_tiles) > 0 and len(wheel3_tiles) > 0)

            on_road = left_side_on_road or right_side_on_road
            
            # Print both lidar and road status
            # print("\rLiDAR Distances: Left 90°: {:<3} | Left 45°: {:<3} | Forward: {:<3} | Right 45°: {:<3} | Right 90°: {:<3} | {}".format(
            #     lidar_readings[90], lidar_readings[45], lidar_readings[0], lidar_readings[-45], lidar_readings[-90],
            #     "On Road" if on_road else "Off Road" 
            # ), end='')
            # print("\rLiDAR Distances: Left 90°: {:<3} | Left 45°: {:<3} | Forward: {:<3} | Right 45°: {:<3} | Right 90°: {:<3} | {}".format(
            #     lidar_readings[175], lidar_readings[120], lidar_readings[90], lidar_readings[60], lidar_readings[5],
            #     "On Road" if on_road else "Off Road" 
            # ), end='')
            # directions = [175, 120, 90, 60, 5]

            if not on_road:
                terminated = True

        frame_count += 1

        if terminated or truncated:
            elapsed_time = time.time() - episode_start_time
            distance = env.episode_distance
            avg_speed = distance / elapsed_time if elapsed_time > 0 else 0

            # Save to CSV
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([episode_counter, distance, elapsed_time, avg_speed, episode_total_reward])

            print(f"\nEpisode {episode_counter} Summary:")
            print(f"  Distance: {distance:.2f} m")
            print(f"  Time: {elapsed_time:.2f} sec")
            print(f"  Avg Speed: {avg_speed:.2f} m/s")
            print(f"  Total Reward: {episode_total_reward:.2f}")

            episode_counter += 1
            episode_total_reward = 0.0
            episode_start_time = time.time()

            observation, info = env.reset(checkpoints=checkpoints.copy())
            print("\nTrack reset!")
            
        clock.tick(60)  # Limit to 60 FPS

    print("\nQuitting...")
    print("\nQuitting...")
    cv2.destroyAllWindows()
    env.close()
    env.close()