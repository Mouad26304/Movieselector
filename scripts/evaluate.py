# evaluate.py

import torch
from movie_trailer_env.custom_env import MovieTrailerEnv
from stable_baselines3 import PPO

def evaluate_model(model, movie_paths, trailer_paths, num_episodes=5):
    """
    Evaluate the trained RL agent on new movie and trailer data.
    
    Args:
        model (PPO): The trained PPO model.
        movie_paths (list): List of paths to the movie videos.
        trailer_paths (list): List of paths to the trailer videos.
        num_episodes (int): Number of episodes to run for evaluation.
    
    Returns:
        mean_reward (float): The average reward over all episodes.
    """
    # Initialize the environment using the first movie (you can loop through more movies if needed)
    env = MovieTrailerEnv(movie_paths[0])
    
    total_reward = 0
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        # Run an episode
        while not done:
            action = model.predict(obs)[0]  # Predict the action using the PPO model
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        
        total_reward += episode_reward
    
    mean_reward = total_reward / num_episodes
    print(f"Mean reward after evaluation: {mean_reward}")
    return mean_reward

if __name__ == "__main__":
    # Example usage (replace these paths with the actual paths to your movies and trailers)
    movie_paths = ["./data/movies/movie_1.mp4", "./data/movies/movie_2.mp4"]
    trailer_paths = ["./data/trailers/movie_1_trailer.mp4", "./data/trailers/movie_2_trailer.mp4"]

    # Load the trained PPO model
    model_path = "./models/ppo_movie_trailer_model"  # Adjust the path where your model is saved
    model = PPO.load(model_path)

    # Evaluate the model on new movie/trailer data
    evaluate_model(model, movie_paths, trailer_paths)
