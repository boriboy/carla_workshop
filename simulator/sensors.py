import sys
import glob
import os
import torch
import globals

import numpy as np
import matplotlib.pyplot as plt

# load carla module
try:
    sys.path.append(glob.glob('../PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

LIDAR_RANGE_M = 50.0  # LiDAR range in meters

class SensorWrapper:
    """
    Base sensor wrapper class.
    """
    data : carla.SensorData = None

    def __init__(self, world):
        """
        Initialize the sensor wrapper.
        """
        self.world = world
        self.ego_vehicle = None
        self.sensor = None
        self._bp = None

    def get_sensor(self):
        return self.sensor
    
    def _on_sensor_data(self, data: carla.SensorData):
        # print(f"Received data from {self.sensor.type_id} at frame {data.frame}")
        self.data = data
        
    def attach_ego(self, vehicle):
        """
        Spawns the sensor (derived from carla.Actor) at ego vehicle transform with rigid attachment.
        
        vehicle: carla.Vehicle - the vehicle to attach the sensor to
        """
        # spawn sensor and attach to ego vehicle
        self.ego_vehicle = vehicle
        self.sensor = self.world.spawn_actor(self._bp, self._transform, 
                                attach_to=self.ego_vehicle,
                                attachment_type=carla.AttachmentType.Rigid)
        
        
    def stop(self):
        """
        Stop listening to sensor data.
        """
        if self.sensor is not None:
            self.sensor.stop()
            
    def start(self):
        """
        Start listening to sensor data.
        """
        if self.sensor is not None:
            self.sensor.listen(self._on_sensor_data)
            
    def get_null_data_tensor(self):
        """ Return a default null tensor for this sensor type. """
        return torch.zeros(self.default_size, dtype=torch.float32).to(globals.device)

class RGBCamera(SensorWrapper):
    default_size = (3, 600, 800)  # CxHxW
    
    """
    Camera sensor wrapper.
    """
    def __init__(self, world, image_size=(800, 600), fov=105):
        super().__init__(world)
        
        # get blueprint and set attributes
        self._bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self._bp.set_attribute("image_size_x", str(image_size[0]))
        self._bp.set_attribute("image_size_y", str(image_size[1]))
        self._bp.set_attribute("fov", str(fov))

        # set transform (relative to vehicle)
        cam_location = carla.Location(2, 0, 1)  # x=2m forward, z=1m up
        cam_rotation = carla.Rotation(0, 0, 0)
        self._transform = carla.Transform(cam_location, cam_rotation)
        
    def tensorize_data(self):
        """ Convert raw image data to a tensor. """
        if self.data is None:
            return None
        img = np.frombuffer(self.data.raw_data, dtype=np.uint8)
        img = img.reshape((self.data.height, self.data.width, 4))

        # Convert to tensor
        img = torch.from_numpy(img).permute(2, 0, 1)  # Change to CxHxW
        return img


class LiDARSensor(SensorWrapper):
    default_size = (1, 256, 256)  # CxHxW
    
    """
    LiDAR sensor wrapper.
    """
    def __init__(self, world, points_per_second=90000, rotation_frequency=40, range=LIDAR_RANGE_M):
        super().__init__(world)

        # get blueprint and set attributes
        self._bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        self._bp.set_attribute('channels', str(32))
        self._bp.set_attribute('points_per_second', str(points_per_second))
        self._bp.set_attribute('rotation_frequency', str(rotation_frequency))
        self._bp.set_attribute('range', str(range))

        # set transform
        lidar_location = carla.Location(0, 0, 2)  # 2m above vehicle center
        lidar_rotation = carla.Rotation(0, 0, 0)
        self._transform = carla.Transform(lidar_location, lidar_rotation)
        
    def tensorize_data(self):
        """ Convert raw LiDAR data to a tensor (e.g., bird's-eye view occupancy grid). """
        if self.data is None:
            return None
        bev = self.lidar_points_to_bev(self.data)
        return torch.tensor(bev, dtype=torch.float32).to(globals.device)
    
    def _on_sensor_data(self, data):
        super()._on_sensor_data(data)
        
        # visualize the BEV for debugging
        # LiDARSensor.carla_bev_to_plot(data, range_m=LIDAR_RANGE_M)
    
        
    @staticmethod
    def lidar_points_to_bev(data):
        """ Convert raw LiDAR data to bird's-eye view occupancy grid. """
        points = LiDARSensor.process_lidar_data(data)
        return LiDARSensor.lidar_to_bev(points)

    @staticmethod
    def lidar_to_bev(points, grid_size=256, range_m=LIDAR_RANGE_M):
        # Filter points within range
        mask = np.sqrt(points[:, 0]**2 + points[:, 1]**2) < range_m
        points = points[mask]
        
        # Convert to grid coordinates
        resolution = range_m * 2 / grid_size
        x_grid = ((points[:, 0] + range_m) / resolution).astype(int)
        y_grid = ((points[:, 1] + range_m) / resolution).astype(int)
        
        # Create occupancy grid
        bev = np.zeros((grid_size, grid_size))
        valid_mask = (x_grid >= 0) & (x_grid < grid_size) & (y_grid >= 0) & (y_grid < grid_size)
        bev[y_grid[valid_mask], x_grid[valid_mask]] = 1
        
        return bev
    
    @staticmethod
    def process_lidar_data(lidar_data):
        # Convert to numpy array
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        
        # Extract x, y, z coordinates (ignore intensity)
        xyz = points[:, :3]
        
        return xyz
    
    @staticmethod
    def carla_bev_to_plot(lidar_data, range_m=50):
        """Convert CARLA LiDAR data directly to BEV plot"""
        # Process CARLA data
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        xyz = points[:, :3]
        
        # Convert to BEV
        bev = LiDARSensor.lidar_to_bev(xyz, grid_size=256, range_m=range_m)
        
        # Visualize
        LiDARSensor.visualize_bev(lidar_data.frame, bev, range_m=range_m, title="CARLA LiDAR BEV")
        
        return bev
    
    @staticmethod
    def visualize_bev(frame, bev_matrix, range_m=50, title="LiDAR Bird's Eye View"):
        """
        Visualize a 256x256 bird's eye view matrix
        
        Args:
            bev_matrix: 256x256 numpy array (0s and 1s for occupancy)
            range_m: detection range in meters
            title: plot title
        """
        plt.figure(figsize=(10, 10))
        
        # Create the plot
        plt.imshow(bev_matrix, 
                cmap='hot',           # Color map (hot, viridis, plasma, etc.)
                origin='lower',       # Origin at bottom-left
                extent=[-range_m, range_m, -range_m, range_m])  # Real-world coordinates
        
        # Add vehicle position (center)
        plt.scatter(0, 0, c='red', s=100, marker='s', label='Vehicle')
        
        # Formatting
        plt.title(title, fontsize=16)
        plt.xlabel('X (meters)', fontsize=12)
        plt.ylabel('Y (meters)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.colorbar(label='Occupancy')
        plt.legend()
        
        # Add range circles for reference
        circle1 = plt.Circle((0, 0), 25, fill=False, color='white', alpha=0.5, linestyle='--')
        circle2 = plt.Circle((0, 0), range_m, fill=False, color='white', alpha=0.5, linestyle='--')
        plt.gca().add_patch(circle1)
        plt.gca().add_patch(circle2)
        
        plt.tight_layout()
        plt.savefig(f"lidar_bev_{frame}.png", bbox_inches='tight')
        plt.close()

class CollisionSensor(SensorWrapper):
    default_size = (1,1,1)  # single binary value
    
    """
    Collision sensor wrapper.
    """
    def __init__(self, world):
        super().__init__(world)

        # get blueprint
        self._bp = self.world.get_blueprint_library().find('sensor.other.collision')

        # set transform
        self._transform = carla.Transform(carla.Location(0, 0, 0))

    def tensorize_data(self):
        """ Convert collision data to a tensor. """
        if self.data is None:
            return None
        return torch.tensor([[1]], dtype=torch.float32).to(globals.device)