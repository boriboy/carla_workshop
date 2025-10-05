from time import time
from random import choice
from rl_model.reward import WaypointRewardFunction
from rl_model.trainer import ModelTrainer
from rl_model.wrapper import ModelWrapper
from simulator.ego import EgoVehicleWrapper
from simulator.interface import *

BATCH_SIZE = 16
MAX_SCENARIO_STEPS = 440  # 22 seconds at 20 FPS
MAX_DISTANCE_TO_GOAL = 50  # 50m
SUCCESS_DISTANCE_TO_GOAL = 0.2  # 20cm

SENSOR_WAIT_S = 0.003  # 3ms


class ScenarioBase:

    _client = None
    _world = None
    _mode = None  # train, test
    _vehicles = []
    _pedestrians = []
    is_running = False
    _config = {}

    # public attributes
    ego_wrapper = None

    def __init__(
        self,
        config,
        client: carla.Client,
        map: carla.Map,
        ego_wrapper: EgoVehicleWrapper,
        junction_id,
        route,
    ):
        self._config = config

        # CARLA client and world
        self._client = client
        self._world = client.get_world()
        self._map = map
        self.ego_wrapper = ego_wrapper
        self._route = route
        self._junction_id = junction_id

        # move ego vehicle to the start of the route
        self.ego_wrapper.move_to_waypoint(self._route[0])

        spectate_mode = config.get("spectate", "top_down")
        if spectate_mode == "top_down":
            # place spectator above the middle of the route
            mid_route_wp = self._route[len(self._route) // 2]
            place_spectator_on_transform(
                self._world, mid_route_wp.transform, height=35.0, pitch=-90.0
            )
        elif spectate_mode == "pov":
            place_spectator_on_transform(
                self._world, self._route[0].transform, height=3.0, pitch=-15.0
            )

        # draw route waypoints (for better understanding of the scenario)
        draw_waypoints(self._world, self._route, life_time=60.0)

    # run scenario with the provided callback
    def run(self, callback):
        # run the provided callback function
        return callback(self)

    # update internal running state
    def set_is_running(self, is_running):
        if is_running:
            print(f"Running scenario")
        else:
            print(f"Stopping scenario")

        self.is_running = is_running

    # cleanup scenario actors
    def cleanup(self):
        # destroy other vehicles
        cleanup_aux_actors(self._client, self._vehicles + self._pedestrians)

        print("Scenario cleaned up")
