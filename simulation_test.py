import gymnasium as gym
import numpy as np
import random

# \venv\Lib\site-packages\gymnasium\envs\box2d\car_racing.py

# the render mode is used to display an image so you can see what is happening, otherwise the simulation will run in the background
# domain_randomize: Changes the physics and appearance of the track randomly

env = gym.make("CarRacing-v3", render_mode="human", domain_randomize=False)  

# observation is the image itself
# info contains metadata (not always used but useful for debugging)
# seed ensures the car appears in the same spot all the time

# seed = random.randint(0, 100000)
observation, info = env.reset(seed=37843)
# print(seed)

# Good maps: 37843, 51070

def is_on_road(pixel):
    # """ Check if a given pixel represents the road (gray color). """
    # return np.all(pixel < [200, 200, 200]) and np.all(pixel > [100, 100, 100])

    """ Check if a pixel falls within the road gray color range. """
    LOWER_GRAY = np.array([100, 100, 100])  # Dark gray
    UPPER_GRAY = np.array([200, 200, 200])  # Light gray

    return np.all(pixel >= LOWER_GRAY) and np.all(pixel <= UPPER_GRAY)

for _ in range(100000): # run onyl 1000 times

    h, w, _ = observation.shape  # Get image dimensions
    y, x = int(h * 0.6), int(w * 0.5)  # Approximate car position in the image
    pixel = observation[y, x]

    ## Check if the pixel is on the road
    # on_road = is_on_road(pixel)
    # print('Is on road?', on_road)

    # action = env.action_space.sample()  # this just picks a random action, for testing purposes (In a real RL algorithm, you would replace this random action with one chosen by your trained policy.)
    # If we only wanted to implement steer and acceleration
    steering = np.random.uniform(-1.0, 1.0)
    acceleration = np.random.uniform(0.0, 1.0)
    brake = np.random.uniform(0.0, 1.0)

    action = np.array([steering, steering, 0])

    # env.step(action) executes the selected action in the environment.
    # observation: The next state (e.g., an image of the new position on the track).
    # reward: A numerical value indicating how good or bad the action was.
    # terminated: True if the episode has ended (e.g., the car crashes or completes the track).
    # truncated: True if the episode is forcefully stopped (e.g., max steps reached).
    # info: Additional information about the step.
    observation, reward, terminated, truncated, info = env.step(action)
   
    # check if the epsido is over
    if terminated or truncated:
        observation, info = env.reset()

# close the environment
env.close()
