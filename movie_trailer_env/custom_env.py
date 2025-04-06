# movie_trailer_env/custom_env.py

import gym
import torch
from movie_trailer_env.env import MovieTrailerEnv
from movie_trailer_env.reward_net import RewardNet

class CustomRewardEnv(MovieTrailerEnv):
    """
    Custom environment that incorporates the learned reward network.
    This subclass overrides the step function to return rewards predicted by the learned RewardNet.
    """
    def __init__(self, video_path, reward_net):
        """
        Initializes the custom environment with the learned reward network.

        Args:
        - video_path (str): Path to the video to be used in the environment.
        - reward_net (RewardNet): The learned reward network.
        """
        super().__init__(video_path)
        self.reward_net = reward_net  # Assign the learned reward network

    def step(self, action):
        """
        Override the `step` function to use the learned reward network instead of a manually defined reward.

        Args:
        - action (int): The action selected by the agent (index of the frame).

        Returns:
        - next_state (np.array): The video frames (state).
        - reward (float): The reward predicted by the learned reward network.
        - done (bool): Whether the episode is done.
        - info (dict): Additional information.
        """
        # Perform the original environment step
        next_state, _, done, info = super().step(action)

        # Extract the state (frame) and action (selected frame)
        state = torch.tensor(self.video[action], dtype=torch.float32).flatten().unsqueeze(0).cuda()
        action_tensor = torch.tensor([action], dtype=torch.float32).unsqueeze(0).cuda()

        # Use the learned reward network to predict the reward for the current state-action pair
        reward = self.reward_net(state, action_tensor).item()

        return next_state, reward, done, info
