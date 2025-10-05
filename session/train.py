from rl_model.reward import WaypointRewardFunction
from rl_model.trainer import ModelTrainer
from rl_model.wrapper import ModelWrapper
from scenario.empty_scenario import EmptyScenario
from scenario.hard_scenario import HardScenario
from scenario.mild_scenario import MildScenario
from scenario.scenario_base import ScenarioBase
from session.base import Session, SessionEnum
from simulator.ego import EgoVehicleWrapper
from simulator.interface import *
from simulator.sensors import CollisionSensor, LiDARSensor

BATCH_SIZE = 16
MAX_SCENARIO_STEPS = 440 # 22 seconds at 20 FPS
MAX_DISTANCE_TO_GOAL = 50 # 50m
SUCCESS_DISTANCE_TO_GOAL = 0.2 # 20cm
SENSOR_WAIT_S = 0.003 # 3ms

class TrainSession(Session):
    def __init__(self, config, client):
        super().__init__(config, client)
        self._mode = SessionEnum.TRAIN        

    def run(self):
        print("Executing train session logic")
        
        # get operating variables
        world = self._client.get_world()            # CARLA world object
        
        # three scenarios to choose from randomly on each training epoch
        scenarios = [EmptyScenario, MildScenario, HardScenario]
        
        # loop epochs
        epochs = self._config.get("epochs", 1)
        
        # enter sync mode
        enter_sync_mode(world, fixed_delta_seconds=0.05)
        
        for epoch in range(epochs):
            print(f"[Train] Epoch {epoch+1}/{epochs}")
            
            # pre-configure scenario parameters
            # select a random junction and a random route in it
            junction_id = choice(list(self._routes.keys()))
            routes = self._routes[junction_id]
            route = choice(routes)
            
            # initiate and run a scenario
            scenario = choice(scenarios)(self._config, self._client, self._map, self.ego_wrapper, junction_id, route)
            
            # start listening to sensors
            self.ego_wrapper.start_all_sensors()
            
            # run the scenario with a callback
            goal_reached, has_collision = scenario.run(self.callback)
            self.summary.add_epoch_result(goal_reached, has_collision)
            
            # stop listening to sensors
            self.ego_wrapper.stop_all_sensors()

            # stop the scenario
            scenario.cleanup()
            
            # save model weights after training
            # saves after every epoch to avoid losing progress in case of a crash on a long training run
            self._model.save_weights()
            
        # exit sync mode
        exit_sync_mode(world)
        
        # destroy ego vehicle
        self.ego_wrapper.destroy()


    def callback(self, scenario: ScenarioBase):
        print(f"[Scenario] starting at junction id {scenario._junction_id}")
        
        scenario.is_running = True
        episode_data = []
        
        # initialize machine learning components
        reward_function = WaypointRewardFunction(scenario._route)
        trainer = ModelTrainer(self._model, learning_rate=1e-4)
        vehicle = scenario.ego_wrapper.get_ego()
        
        # get first frame of the scenario
        start_frame = scenario._world.tick()
        
        while (scenario.is_running):
            # advance the world to next frame
            frame = scenario._world.tick()
            
            # forward pass
            try:
                vehicle_control, has_collision, frame_data = self._model.forward_pass(scenario.ego_wrapper,
                                                                                frame, scenario._route[-1],
                                                                                reward_function,
                                                                                sensor_data_wait_s=SENSOR_WAIT_S)
            except (KeyError, ValueError) as e:
                # an unexpected error occurred within forward pass
                # this requires more investigation
                print(f"[Error] during forward pass: {e}")
                continue
            
            # append summarized frame data
            episode_data.append(frame_data)
            
            # apply the control to the ego vehicle
            vehicle.apply_control(vehicle_control)
            
            # was batch_size reached?
            batch_size_reached = len(episode_data) >= BATCH_SIZE
            
            # back propagation (trains the model on the batch)
            if batch_size_reached:
                self._model.back_prop(trainer, episode_data)
                episode_data = []  # clear after training
            
            # evaluate scenario termination
            # scenario termination criteria:
            # 1. collision
            # 2. max steps reached
            # 3. distance to final waypoint too high
            # 4. goal reached
            distance_to_goal_m = vehicle.get_location().distance(scenario._route[-1].transform.location)
            reached_max_steps = (frame - start_frame) >= MAX_SCENARIO_STEPS
            goal_reached = distance_to_goal_m <= SUCCESS_DISTANCE_TO_GOAL # within 20cm of the goal
            should_terminate = has_collision or reached_max_steps or distance_to_goal_m > MAX_DISTANCE_TO_GOAL or goal_reached

            # terminate scenario if any of the criteria is met
            if should_terminate:
                self._model.back_prop(trainer, episode_data)
                episode_data = []  # clear after training
                scenario.set_is_running(False)
                print(f"Termination criteria met: " +
                      (f"collision " if has_collision else "") +
                      (f"max steps reached " if reached_max_steps else "") +
                      (f"distance to goal too high " if distance_to_goal_m > MAX_DISTANCE_TO_GOAL else "") +
                      (f"goal reached " if goal_reached else ""))
                print(f"Terminating scenario.")
                continue

        print(f"[Scenario] ended.")
        return goal_reached, has_collision  # return whether the scenario was successful (goal reached)