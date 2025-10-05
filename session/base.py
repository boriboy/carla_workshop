from abc import ABC, abstractmethod

from rl_model.wrapper import ModelWrapper
from simulator.ego import EgoVehicleWrapper
from simulator.interface import get_junction_routes
from simulator.sensors import CollisionSensor, LiDARSensor


class SessionEnum:
    TRAIN = "train"
    TEST = "test"


class Session(ABC):
    """
    Session base class, defines an abstract runtime session.
    Currently supported: train, test
    """

    _mode = None  # train, test
    _client = None  # carla client object
    _map = None  # carla map object

    def __init__(self, config, carla_client):
        # runtime configuration
        self._config = config
        self._client = carla_client
        self.summary = SessionSummary(config.get("epochs", 1))

        # getting the map object is expensive therefore we call it once and save it
        self._map = carla_client.get_world().get_map()

        self._model = ModelWrapper(self._config)  # get wrapped model

        # get all possible routes in the map junctions
        self._routes = get_junction_routes(self._map)

        # initiate ego vehicle and attach sensors
        self.ego_wrapper = EgoVehicleWrapper(config, carla_client, self._map)
        self.ego_wrapper.attach_sensor(LiDARSensor(carla_client.get_world()))
        self.ego_wrapper.attach_sensor(CollisionSensor(carla_client.get_world()))

    @abstractmethod
    def run(self):
        pass


class SessionSummary:
    """
    Class to summarize the results of a session
    Currently very basic, just counts successful epochs
    """

    def __init__(self, planned_num_epochs):
        self.success_count = 0
        self.collision_count = 0
        self.actual_epochs = 0
        self.planned_epochs = planned_num_epochs

    def add_epoch_result(self, goal_reached, has_collision):
        """Add the result of an epoch to the summary"""
        if goal_reached:
            self.success_count += 1

        if has_collision:
            self.collision_count += 1

        self.actual_epochs += 1

    def get_success_rate(self):
        """Get the success rate of the session"""
        return (
            self.success_count / self.actual_epochs if self.actual_epochs > 0 else 0.0
        )

    def __str__(self):
        return f"Session Summary: {self.success_count}/{self.actual_epochs} successful epochs (Planned: {self.planned_epochs}) - Success Rate: {self.get_success_rate()*100:.2f}% - Collision Count: {self.collision_count}"
