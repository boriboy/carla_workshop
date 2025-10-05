import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import globals
from .wrapper import ModelWrapper

class ModelTrainer:
    def __init__(self, model: ModelWrapper, learning_rate=1e-4, buffer_size=1000, 
                 gamma=0.99, value_loss_coef=0.5, entropy_coef=0.01):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.experience_buffer = deque(maxlen=buffer_size)
        
        # PPO/Actor-Critic hyperparameters
        self.gamma = gamma  # Discount factor
        self.value_loss_coef = value_loss_coef  # Value loss coefficient
        self.entropy_coef = entropy_coef  # Entropy regularization coefficient
        
    def train_on_batch(self, episode_data):
        """
        Train the model on a batch of episode data using Actor-Critic/PPO approach
        
        Args:
            episode_data: List of dictionaries containing:
                - 'state_lidar': LiDAR sensor data
                - 'state_collision': Collision sensor data
                - 'action': Actions taken
                - 'reward': Rewards received
                - 'value': Value estimates from the model
                - 'action_log_prob': Log probabilities of actions
        """
        if len(episode_data) == 0:
            return
            
        # Convert episode data to tensors
        states_lidar = torch.stack([exp['state_lidar'] for exp in episode_data])
        states_collision = torch.stack([exp['state_collision'] for exp in episode_data])
        states_relative_position = torch.stack([exp['state_relative'] for exp in episode_data])
        actions = torch.stack([exp['action'] for exp in episode_data])
        rewards = torch.tensor([exp['reward'] for exp in episode_data], dtype=torch.float32)
        old_values = torch.stack([exp['value'] for exp in episode_data]).squeeze()
        old_log_probs = torch.stack([exp['action_log_prob'] for exp in episode_data])
        
        # Move to device if using GPU
        if hasattr(globals, 'device'):
            states_lidar = states_lidar.to(globals.device)
            states_collision = states_collision.to(globals.device)
            actions = actions.to(globals.device)
            rewards = rewards.to(globals.device)
            old_values = old_values.to(globals.device)
            old_log_probs = old_log_probs.to(globals.device)
        
        # Calculate returns and advantages using GAE (Generalized Advantage Estimation)
        returns, advantages = self._compute_returns_and_advantages(rewards, old_values)
        
        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Forward pass through current model
        self.model.train()
        action_means, action_stds, values = self.model(states_lidar, states_collision, states_relative_position)
        
        # Calculate new action log probabilities
        distributions = torch.distributions.Normal(action_means, action_stds)
        new_log_probs = distributions.log_prob(actions).sum(dim=-1)
        
        # Calculate entropy for exploration bonus
        entropy = distributions.entropy().sum(dim=-1).mean()
        
        # PPO policy loss with clipping
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages  # 0.2 is clip_param
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value function loss
        values = values.squeeze()
        value_loss = F.mse_loss(values, returns)
        
        # Total loss
        total_loss = (policy_loss + 
                     self.value_loss_coef * value_loss - 
                     self.entropy_coef * entropy)
        
        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Log training metrics
        print(f"Training - Total Loss: {total_loss.item():.4f}, "
              f"Policy Loss: {policy_loss.item():.4f}, "
              f"Value Loss: {value_loss.item():.4f}, "
              f"Entropy: {entropy.item():.4f}")
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }
    
    def _compute_returns_and_advantages(self, rewards, values, last_value=0.0):
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: Tensor of rewards
            values: Tensor of value estimates
            last_value: Value estimate for the last state (default 0 for terminal state)
        
        Returns:
            returns: Discounted returns
            advantages: GAE advantages
        """
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        # Convert to numpy for easier computation
        rewards_np = rewards.detach().cpu().numpy()
        values_np = values.detach().cpu().numpy()
        
        # Compute returns (discounted rewards)
        running_return = last_value
        for t in reversed(range(len(rewards))):
            running_return = rewards_np[t] + self.gamma * running_return
            returns[t] = running_return
        
        # Compute advantages using GAE
        gae_lambda = 0.95  # GAE lambda parameter
        running_advantage = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values_np[t + 1]
                
            # Handle case where values_np is a scalar
            if values_np.shape is ():
                values_np = values_np.reshape(-1)

            delta = rewards_np[t] + self.gamma * next_value - values_np[t]
            running_advantage = delta + self.gamma * gae_lambda * running_advantage
            advantages[t] = running_advantage
        
        return returns, advantages
    
    def train_step(self, state, action, reward, next_state):
        """Store experience and perform training (legacy method, kept for compatibility)"""
        # Store experience
        self.experience_buffer.append((state, action, reward, next_state))
        
        # Train if we have enough samples
        if len(self.experience_buffer) >= 32:
            self._train_batch()
    
    def _train_batch(self, batch_size=32):
        """Perform backpropagation on a batch (legacy method, kept for compatibility)"""
        # Sample batch
        batch = list(self.experience_buffer)[-batch_size:]
        
        states = torch.cat([exp[0]['rgb'] for exp in batch])
        rewards = torch.tensor([exp[2] for exp in batch]).float()
        
        if hasattr(globals, 'device'):
            states = states.to(globals.device)
            rewards = rewards.to(globals.device)
        
        # Forward pass
        self.model.train()
        outputs = self.model(states)
        
        # Calculate loss (example: reward prediction)
        predicted_rewards = outputs.get('reward_pred', outputs['throttle'])
        loss = nn.MSELoss()(predicted_rewards.squeeze(), rewards)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        print(f"Training loss: {loss.item():.4f}")