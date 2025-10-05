import numpy as np

class WaypointRewardFunction:
    def __init__(self, waypoints, completion_reward=100, collision_penalty=-200):
        self.waypoints = waypoints
        self.current_waypoint_idx = 0
        self.completion_reward = completion_reward
        self.collision_penalty = collision_penalty
        self.last_progress = 0
        
    def calculate_reward(self, vehicle_location, vehicle_velocity, collision_detected):
        """
        Calculate reward based on waypoint following, progress, and safety
        
        Args:
            vehicle_location: carla.Location of the vehicle
            vehicle_velocity: carla.Vector3D of vehicle velocity  
            collision_detected: bool from collision sensor
        
        Returns:
            float: total reward for current step
        """
        total_reward = 0
        
        # 0.1 Always punish by distance in meters to goal (last waypoint)
        total_reward -= vehicle_location.distance(self.waypoints[-1].transform.location) * 0.1
        
        # 1. COLLISION PENALTY (highest priority)
        if collision_detected:
            return self.collision_penalty
        
        # 2. ROUTE PROGRESS REWARD
        progress_reward = self._calculate_progress_reward(vehicle_location)
        total_reward += progress_reward
        
        # 2.1 Penalty for no progress
        if progress_reward < 0:
            total_reward += progress_reward  # double penalty for regression
        
        # 3. CROSS-TRACK ERROR PENALTY  
        deviation_penalty = self._calculate_deviation_penalty(vehicle_location)
        total_reward += deviation_penalty
        
        # 4. HEADING ALIGNMENT REWARD
        heading_reward = self._calculate_heading_reward(vehicle_location, vehicle_velocity)
        total_reward += heading_reward
        
        # 5. SPEED REWARD
        speed_reward = self._calculate_speed_reward(vehicle_velocity)
        total_reward += speed_reward
        
        # 6. COMPLETION BONUS
        completion_bonus = self._check_completion_bonus(vehicle_location)
        total_reward += completion_bonus
        
        return total_reward
    
    def _calculate_progress_reward(self, vehicle_location):
        """Reward forward progress through waypoint sequence"""
        if self.current_waypoint_idx >= len(self.waypoints):
            return 0
            
        # Distance to current target waypoint
        target_waypoint = self.waypoints[self.current_waypoint_idx]
        distance_to_target = vehicle_location.distance(target_waypoint.transform.location)
        
        # Check if close enough to advance to next waypoint
        if distance_to_target < 3.0:  # 3 meter threshold
            self.current_waypoint_idx += 1
            return 20  # Waypoint reached bonus
            
        # Progress reward based on getting closer to target
        if self.current_waypoint_idx > 0:
            prev_waypoint = self.waypoints[self.current_waypoint_idx - 1]
            total_segment_distance = target_waypoint.transform.location.distance(
                prev_waypoint.transform.location)
            distance_from_prev = vehicle_location.distance(prev_waypoint.transform.location)
            progress = min(distance_from_prev / total_segment_distance, 1.0)
            
            progress_reward = (progress - self.last_progress) * 10
            self.last_progress = progress
            return progress_reward
        
        return 0
    
    def _calculate_deviation_penalty(self, vehicle_location):
        """Penalize lateral deviation from planned route"""
        if self.current_waypoint_idx >= len(self.waypoints):
            return 0
            
        # Find closest point on route segment
        if self.current_waypoint_idx == 0:
            closest_distance = vehicle_location.distance(
                self.waypoints[0].transform.location)
        else:
            # Distance to line segment between consecutive waypoints
            p1 = self.waypoints[self.current_waypoint_idx - 1].transform.location
            p2 = self.waypoints[self.current_waypoint_idx].transform.location
            closest_distance = self._point_to_line_distance(vehicle_location, p1, p2)
        
        # Exponential penalty for deviation
        if closest_distance < 2.0:
            return 0  # No penalty within 2m
        elif closest_distance < 5.0:
            return -2 * (closest_distance - 2.0)  # Linear penalty 2-5m
        else:
            return -10 * np.exp(closest_distance - 5.0)  # Exponential penalty >5m
    
    def _calculate_heading_reward(self, vehicle_location, vehicle_velocity):
        """Reward alignment with desired heading toward next waypoint"""
        if self.current_waypoint_idx >= len(self.waypoints):
            return 0
            
        # Vehicle heading vector
        velocity_magnitude = np.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2)
        if velocity_magnitude < 0.1:  # Stationary vehicle
            return 0
            
        vehicle_heading = np.array([vehicle_velocity.x, vehicle_velocity.y]) / velocity_magnitude
        
        # Desired heading toward next waypoint
        target_location = self.waypoints[self.current_waypoint_idx].transform.location
        desired_direction = np.array([
            target_location.x - vehicle_location.x,
            target_location.y - vehicle_location.y
        ])
        desired_magnitude = np.linalg.norm(desired_direction)
        
        if desired_magnitude < 0.1:
            return 0
            
        desired_heading = desired_direction / desired_magnitude
        
        # Dot product gives alignment (-1 to 1)
        alignment = np.dot(vehicle_heading, desired_heading)
        
        # Convert to reward (0 to 5)
        return 2.5 * (alignment + 1)
    
    def _calculate_speed_reward(self, vehicle_velocity):
        """Reward appropriate speed for junction navigation"""
        speed_kmh = 3.6 * np.sqrt(
            vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2)
        
        # Optimal speed range for junction turns: 10-25 km/h
        if 10 <= speed_kmh <= 25:
            return 10.0
        elif 5 <= speed_kmh < 10:
            return 5.0  # Slightly slow but safe
        elif 25 < speed_kmh <= 35:
            return -5.0  # Slightly fast
        elif speed_kmh < 5:
            return -(5**2 - speed_kmh**2)  # Too slow, blocking traffic
        else:
            return -15.0  # Dangerously fast

    def _check_completion_bonus(self, vehicle_location):
        """Large bonus for completing the entire route"""
        if self.current_waypoint_idx >= len(self.waypoints):
            # Check if we're close to the final waypoint
            final_waypoint = self.waypoints[-1]
            distance_to_final = vehicle_location.distance(final_waypoint.transform.location)
            if distance_to_final < 5.0:
                return self.completion_reward
        return 0
    
    def _point_to_line_distance(self, point, line_start, line_end):
        """Calculate perpendicular distance from point to line segment"""
        # Vector from line_start to line_end
        line_vec = np.array([line_end.x - line_start.x, line_end.y - line_start.y])
        line_length_sq = np.dot(line_vec, line_vec)
        
        if line_length_sq < 1e-6:  # Degenerate line
            return point.distance(line_start)
        
        # Vector from line_start to point
        point_vec = np.array([point.x - line_start.x, point.y - line_start.y])
        
        # Project point onto line
        t = max(0, min(1, np.dot(point_vec, line_vec) / line_length_sq))
        
        # Closest point on line segment
        closest_point = np.array([line_start.x, line_start.y]) + t * line_vec
        
        # Distance from point to closest point on line
        distance_vec = np.array([point.x, point.y]) - closest_point
        return np.linalg.norm(distance_vec)
    
    def reset(self):
        """Reset for new episode"""
        self.current_waypoint_idx = 0
        self.last_progress = 0

