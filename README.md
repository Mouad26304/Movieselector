Movie Trailer Selector Project Files
This project uses reinforcement learning (RL) and inverse reinforcement learning (IRL) techniques to select key scenes from movies for generating trailers. Below is an explanation of the functionality of each file in the project.


data/
movies/: Contains the raw movie video files. Each movie is named according to the format movie_X.mp4, where X is the movie index.

trailers/: Contains the corresponding trailer video files for the movies, named in the format movie_X_trailer.mp4.

movie_trailer_env/
__init__.py: Initializes the movie_trailer_env package.

env.py: Defines the MovieTrailerEnv class, which represents the environment where the RL agent interacts with the movie data. The agent selects key frames from the movie based on a reward model.

expert_dataset.py: Contains the ExpertDataset class, which loads expert demonstrations (trajectories) to train the reward network. It maps the movie and trailer paths to the corresponding frames for training.

reward_net.py: Defines the RewardNet class, which implements the reward network used for MaxEnt IRL. This network is used to learn rewards based on the expert trajectories.

custom_env.py: Defines a custom environment using the learned reward model, which is used during RL training.

scripts/
train_reward.py: Contains the training loop for the reward network using Maximum Entropy IRL (MaxEnt IRL). It uses the expert dataset and environment to train the reward model, which helps the RL agent evaluate the quality of scenes for trailer selection.

train_rl.py: Contains the training loop for the Proximal Policy Optimization (PPO) algorithm. This script trains the RL agent to select key scenes from the movie for generating a trailer based on the reward network learned in train_reward.py.

evaluate.py: Used to evaluate the trained RL agent on new, unseen videos to see how well it can select scenes for trailers.

tests/
__init__.py: Initializes the tests package.

test_env.py: Contains unit tests for the MovieTrailerEnv class and other environment-related functionalities. Ensures that the environment works as expected.

requirements.txt: Lists the Python dependencies needed to run the project. This includes packages like gym, torch, longvu, and others required for the RL training and video processing.
