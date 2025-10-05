import torch
import rl_model
from rl_model.model import JunctionTurnArchitecture
from simulator.interface import actions_to_control, get_vehicle_control_object
import numpy as np
import math
import globals
import time

MODELS_PATH: str = "data/models"

class ModelWrapper(JunctionTurnArchitecture):
    def __init__(self, config):
        super().__init__()
        
        # store config
        self.config = config
        
        # load pre-trained weights if exists
        self.load_weights()
        
    # single step forward pass, returns control, collision status, and frame data
    def forward_pass(self, ego_wrapper, frame: int, final_destination_wp, reward_function, sensor_data_wait_s = 0.003):
        vehicle = ego_wrapper.get_ego()
        data = ego_wrapper.get_frame_sensor_data(frame)
        
        # conditional sleep to wait for sensor data
        if sensor_data_wait_s > 0:
            time.sleep(sensor_data_wait_s) # ensure sensor data is ready
            
        relative_position_norm, relative_position = ModelWrapper.preprocess_relative_location_and_yaw(
            ego_wrapper.get_ego().get_transform(),
            final_destination_wp.transform
        )

        # forward pass through the model
        action_mean, action_std, value = self(
            data["sensor.lidar.ray_cast"].unsqueeze(0),
            data["sensor.other.collision"].unsqueeze(0),
            relative_position_norm.unsqueeze(0)
        )
        
        # convert model output to a sample of action-space distribution
        action, log_prob = ModelWrapper.sample_action(action_mean, action_std)

        # boolean indicating whether a collision has occurred
        has_collision = data["sensor.other.collision"].flatten().bool().item()
        
        # translate action to vehicle control
        vehicle_control = actions_to_control(action)
        
        # compute reward
        reward = reward_function.calculate_reward(
            vehicle.get_location(),
            vehicle.get_velocity(), 
            has_collision
        )
        
        # store experience
        frame_data = {
            'state_lidar': data["sensor.lidar.ray_cast"],
            'state_collision': data["sensor.other.collision"],
            'state_relative': relative_position_norm,
            'action': action_mean,
            'reward': reward,
            'value': value,
            'action_log_prob': log_prob
        }
        
        return vehicle_control, has_collision, frame_data
    
    # perform back-propagation and training step
    def back_prop(self, trainer, episode_data):
        # invoke training on the batch
        trainer.train_on_batch(episode_data)
        return
        
    def load_weights(self):
        """Load model weights from disk if available"""
        model_path = self._get_model_path()
        try:
            self.load_state_dict(torch.load(model_path))
            print(f"Loaded weights from {model_path}")
        except FileNotFoundError:
            print(f"No pre-trained model found at {model_path}")
            
    def save_weights(self):
        """Save model weights to disk"""
        model_path = self._get_model_path()
        torch.save(self.state_dict(), model_path)
        print(f"Saved weights to {model_path}")
        
    def _get_model_path(self):
        model_name = self.config.get("model_name", "default_model")
        return f"{MODELS_PATH}/{model_name}.pt"
    
    @staticmethod
    def actions_to_control(model_output):
        """Convert model output dictionary to carla.VehicleControl"""
        return actions_to_control(model_output[0].flatten())
    
    @staticmethod
    def sample_action(action_mean, action_std):
        """
        Sample actions from a normal distribution for exploration
        
        Args:
            action_mean: torch.Tensor, shape (action_dim,) - mean actions
            action_std: torch.Tensor, shape (action_dim,) - standard deviations
        
        Returns:
            action: torch.Tensor, sampled action values
            log_prob: torch.Tensor, log probability of sampled action
        """
        # Create normal distribution
        try:
            distribution = torch.distributions.Normal(action_mean, action_std)
        except ValueError as e:
            # at times when error is too big model outputs are NaN
            raise
        
        # Sample from the distribution
        action = distribution.sample()
        
        # Calculate log probability (needed for training)
        log_prob = distribution.log_prob(action).sum(dim=-1)
        
        return action, log_prob
    
    @staticmethod
    def preprocess_relative_location_and_yaw(vehicle_transform, waypoint_transform, max_range=50.0):
        def wrap_to_pi(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi
        
        # Positions
        veh_x, veh_y = vehicle_transform.location.x, vehicle_transform.location.y
        wp_x, wp_y = waypoint_transform.location.x, waypoint_transform.location.y

        dx = wp_x - veh_x
        dy = wp_y - veh_y

        # Vehicle heading (yaw in radians)
        veh_yaw = math.radians(vehicle_transform.rotation.yaw)
        wp_yaw = math.radians(waypoint_transform.rotation.yaw)

        # Rotate to vehicle-local frame
        local_dx =  math.cos(-veh_yaw) * dx - math.sin(-veh_yaw) * dy
        local_dy =  math.sin(-veh_yaw) * dx + math.cos(-veh_yaw) * dy

        # Normalize position
        norm_dx = np.clip(local_dx / max_range, -1.0, 1.0)
        norm_dy = np.clip(local_dy / max_range, -1.0, 1.0)

        # Relative heading
        delta_yaw = wrap_to_pi(wp_yaw - veh_yaw)
        norm_dyaw = delta_yaw / np.pi  # range [-1, 1]
        
        # relative position
        relative_position_norm = torch.tensor([norm_dx, norm_dy, norm_dyaw], dtype=torch.float32).to(globals.device)
        relative_position = torch.tensor([local_dx, local_dy, delta_yaw], dtype=torch.float32).to(globals.device)
        return relative_position_norm, relative_position