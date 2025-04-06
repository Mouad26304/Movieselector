# movie_trailer_env/__init__.py

"""
MovieTrailerEnv Package
========================
This module provides the custom environment for movie trailer generation using Reinforcement Learning (RL) and Inverse Reinforcement Learning (IRL).
The environment leverages the LongVU model to dynamically assess scenes and provide rewards based on their suitability for a movie trailer.
"""

from .env import MovieTrailerEnv
from .expert_dataset import ExpertDataset
from .reward_net import RewardNet
from .custom_env import CustomRewardEnv

__all__ = [
    "MovieTrailerEnv",  # Base environment
    "ExpertDataset",    # Expert data for IRL
    "RewardNet",        # Reward network used in IRL
    "CustomRewardEnv",  # Custom environment using learned reward
]
