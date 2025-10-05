import torch
import torch.nn as nn
import torch.nn.functional as F

class JunctionTurnArchitecture(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN backbone for 256x256 LiDAR input
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 128x128
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4)  # 4x4
        )
        
        # Process collision sensor (binary input)
        self.collision_fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )
        
        # Process relative position (x, y, yaw)
        self.relative_position_fc = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )

        # Shared feature layer (CNN features + collision features)
        feature_size = 256 * 4 * 4 + 32 + 32  # CNN output + collision features + relative position features
        self.shared_fc = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Actor head
        self.actor_mean = nn.Linear(256, 3) # [steer, throttle, brake]
        self.actor_std = nn.Linear(256, 3)  # [steer, throttle, brake]

        # Critic head
        self.critic = nn.Linear(256, 1)

    def clip_gradients(self, max_norm):
        """Clips gradients to prevent exploding gradients."""
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)

    def forward(self, lidar_state, collision_state, relative_position):
        # Process LiDAR: (batch, 1, 256, 256)
        conv_out = self.conv_layers(lidar_state)
        lidar_features = conv_out.view(conv_out.size(0), -1)
        
        # Process collision: (batch, 1) -> binary float
        collision_features = self.collision_fc(collision_state.float())
        
        # Relative position processing (batch, 3)
        relative_features = self.relative_position_fc(relative_position)
        # Combine all features
        # combined_features = torch.cat((lidar_features, collision_features, relative_features), dim=1)
        
        # Concatenate features
        combined_features = torch.cat((lidar_features.flatten(start_dim=1), collision_features.flatten(start_dim=1),
                                       relative_features.flatten(start_dim=1)), dim=1)
        features = self.shared_fc(combined_features)
        
        action_mean = torch.tanh(self.actor_mean(features))
        action_std = F.softplus(self.actor_std(features))
        state_value = self.critic(features)
        
        return action_mean, action_std, state_value