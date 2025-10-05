import sys
import glob
import os
import globals
import torch
from .interface import *
from .sensors import SensorWrapper

# load carla module
try:
    sys.path.append(glob.glob('../PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# ==============================================================================

class EgoVehicleWrapper:
    sensors = [] # list(SensorWrapper)
    
    """
    Wrapper for the ego vehicle in the CARLA simulator.
    """
    def __init__(self, config: dict, client: carla.Client, map: carla.Map):
        """
        Create an ego vehicle in the CARLA world based on the provided configuration.
        Provides methods to access and control the ego vehicle.
        """
        self._config = config
        self._client = client
        self._map = map
        self.ego = None

        # ego blueprint from config
        ego_bp_name = config.get("vehicle_model", "vehicle.tesla.model3")
        ego_bp = get_blueprint_by_name(client.get_world(), ego_bp_name)
        
        # spawn ego vehicle at a random spawn point
        spawn_point = choice(self._map.get_spawn_points())
        self.ego = spawn_ego_on_transform(client.get_world(), ego_bp, spawn_point)
        
    def get_ego(self):
        """
        Get the CARLA ego vehicle actor.
        """
        return self.ego
    
    def destroy(self):
        """
        Destroy the ego vehicle and all attached sensors.
        """
        for sensor in self.sensors:
            sensor.get_sensor().destroy()
        if self.ego is not None:
            self.ego.destroy()
            self.ego = None
            print("[EgoVehicleWrapper] Ego vehicle destroyed.")
    
    def move_to_waypoint(self, waypoint: carla.Waypoint):
        """
        Move the ego vehicle to a specific waypoint.
        """
        self.ego.set_transform(waypoint.transform)
        
    def attach_sensor(self, sensor_wrapper: SensorWrapper):
        """
        Attach a sensor to the ego vehicle.
        """
        # instruct world to spawn the sensor
        sensor_wrapper.attach_ego(self.ego)
        
        # append to ego's sensors list
        self.sensors.append(sensor_wrapper)
    
    def stop_all_sensors(self):
        """
        Stop listening to all attached sensors.
        """
        for sensor in self.sensors:
            sensor.stop()

    def start_all_sensors(self):
        """
        Start listening to all attached sensors.
        """
        for sensor in self.sensors:
            sensor.start()
            
    def get_frame_sensor_data(self, frame: int):
        """
        Get the latest data from all sensors for a specific frame.
        Returns a dictionary mapping sensor types to their data.
        """
        # [todo] can be more readable
        data = {}
        for sensor in self.sensors:
            # default null data tensor
            data[sensor.sensor.type_id] = sensor.get_null_data_tensor()
            
            if sensor.data is None:
                # if no data received yet, return a zero tensor
                data[sensor.sensor.type_id] = torch.tensor([0], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(globals.device)
                
            elif sensor.data is not None and sensor.data.frame == frame:
                data[sensor.sensor.type_id] = sensor.tensorize_data().unsqueeze(0)
        return data