from .scenario_base import ScenarioBase
from simulator.interface import spawn_pedestrians_around_wp, spawn_vehicles_around_wp

NUM_VEHICLES = 10
NUM_PEDESTRIANS = 10

class HardScenario(ScenarioBase):
    def __init__(self, config, client, map, ego_wrapper, junction_id, route):
        # init ego position, route, spectator positioning, etc.
        super().__init__(config, client, map, ego_wrapper, junction_id, route)
        
        # get mid-route waypoint
        mid_route_wp = self._route[len(self._route)//2]
        
        # get traffic lights ignore setting
        ignore_traffic_lights = config.get("tlights_ignore", True)
        
        # spawn few vehicles and pedestrians with hard congestion
        self._vehicles = spawn_vehicles_around_wp(client, mid_route_wp.transform.location, ignore_traffic_lights, num_vehicles=NUM_VEHICLES
                                                    , search_distance=30.0)
        self._pedestrians = spawn_pedestrians_around_wp(client, mid_route_wp.transform.location, num_pedestrians=NUM_PEDESTRIANS
                                                    , radius=30.0)
        
        print("[HardScenario] initialized")