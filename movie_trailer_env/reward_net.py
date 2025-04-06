# movie_trailer_env/reward_net.py

import torch
import torch.nn as nn
import torch.optim as optim

class RewardNet(nn.Module):
    """
    A simple neural network to estimate the reward function from state-action pairs.
    The network takes the concatenated state-action pair as input and outputs the estimated reward.
    """
    def __init__(self, state_dim, action_dim):
        super(RewardNet, self).__init__()
        
        # Define a fully connected neural network to approximate the reward function
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),  # Input: state + action
            nn.ReLU(),  # ReLU activation
            nn.Linear(256, 1)  # Output: Reward (single scalar value)
        )
        
    def forward(self, state, action):
        """
        Forward pass through the network.
        Concatenate the state and action, then pass through the network.
        """
        x = torch.cat([state, action], dim=-1)  # Concatenate state and action
        return self.net(x)


def train_reward(expert_dataset, env, epochs=100):
    """
    Train the reward network using expert trajectories (state-action pairs) and MaxEnt IRL.
    """
    reward_net = RewardNet(state_dim=env.observation_space.shape[1], action_dim=1).to('cuda')
    optimizer = optim.Adam(reward_net.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        expert_sas = []  # List to store state-action pairs from expert data
        
        # Collect state-action pairs from the expert dataset
        for traj in expert_dataset.trajectories:
            for action in traj:
                state = torch.tensor(env.video[action], dtype=torch.float32).flatten().unsqueeze(0).cuda()
                action_tensor = torch.tensor([action], dtype=torch.float32).unsqueeze(0).cuda()
                expert_sas.append((state, action_tensor))

        # Compute expert rewards
        expert_rewards = torch.stack([reward_net(s, a) for s, a in expert_sas])

        # Generate random policy actions to simulate random trajectory
        rand_actions = torch.randint(0, env.action_space.n, (len(expert_sas), 1)).float().cuda()
        rand_states = torch.stack([torch.tensor(env.video[action], dtype=torch.float32).flatten().cuda() for action in rand_actions.squeeze()])

        # Calculate random rewards
        rand_rewards = reward_net(rand_states, rand_actions)

        # MaxEnt IRL Loss = Expert reward - log_sum_exp(Random reward)
        loss = -(expert_rewards.mean() - torch.logsumexp(rand_rewards, dim=0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return reward_net
