import gymnasium as gym
import numpy as np
import cv2
import pygame

class ManualCarSim(gym.Wrapper):
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
    
    SEED = 37843
    env = ManualCarSim(seed=SEED, show_lidar=True)  # Start with LiDAR visualization enabled
    observation, info = env.reset()

    print("Controls:")
    print("- Arrow keys: Steering (left/right) and Gas (up)")
    print("- Space or Down Arrow: Brake")
    print("- L: Toggle LiDAR visualization")
    print("- Q: Quit")
    
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
            print("\rLiDAR Distances: Left 90°: {:<3} | Left 45°: {:<3} | Forward: {:<3} | Right 45°: {:<3} | Right 90°: {:<3} | {}".format(
                lidar_readings[90], lidar_readings[45], lidar_readings[0], lidar_readings[-45], lidar_readings[-90],
                "On Road" if on_road else "Off Road"
            ), end='')
        frame_count += 1

        if terminated or truncated:
            observation, info = env.reset()
            print("\nTrack reset!")
            
        clock.tick(60)  # Limit to 60 FPS

    print("\nQuitting...")
    env.close()
