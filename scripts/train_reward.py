# train_reward.py

import torch
import torch.optim as optim
from movie_trailer_env.expert_dataset import ExpertDataset
from movie_trailer_env.reward_net import RewardNet
import numpy as np

def train_reward(expert_dataset, env, epochs=100, batch_size=32, lr=1e-3):
    """
    Train the reward network using Maximum Entropy IRL (MaxEnt IRL).
    
    Args:
        expert_dataset (ExpertDataset): The dataset containing expert trajectories.
        env (gym.Env): The environment to interact with during training.
        epochs (int): Number of epochs to train the reward network.
        batch_size (int): Number of samples to process in each training batch.
        lr (float): Learning rate for the optimizer.
    
    Returns:
        reward_net (RewardNet): The trained reward network.
    """
    
    # Initialize the reward network
    reward_net = RewardNet(state_dim=env.observation_space.shape[1], action_dim=1).to('cuda')
    optimizer = optim.Adam(reward_net.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        
        # Sample expert state-action pairs in batches
        batch_sas = []
        for traj in expert_dataset.trajectories:
            for action in traj:
                state = torch.tensor(env.video[action], dtype=torch.float32).flatten().unsqueeze(0).cuda()
                action_tensor = torch.tensor([action], dtype=torch.float32).unsqueeze(0).cuda()
                batch_sas.append((state, action_tensor))
        
        # Shuffle the state-action pairs
        np.random.shuffle(batch_sas)

        # Process the data in batches
        for i in range(0, len(batch_sas), batch_size):
            batch = batch_sas[i:i+batch_size]
            states = torch.cat([s for s, _ in batch], dim=0)
            actions = torch.cat([a for _, a in batch], dim=0)
            
            # Get the rewards predicted by the network
            rewards = reward_net(states, actions)
            
            # Compute the loss (MaxEnt IRL style loss)
            # Here we just use the reward itself, but you can extend this with more sophisticated loss functions
            loss = -rewards.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print loss every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss}")

    return reward_net
