from rl_model.reward import WaypointRewardFunction
from scenario.scenario_base import ScenarioBase
from scenario.empty_scenario import EmptyScenario
from scenario.hard_scenario import HardScenario
from scenario.mild_scenario import MildScenario
from session.base import Session, SessionEnum
from simulator.interface import *

MAX_SCENARIO_STEPS = 880 # 44 seconds at 20 FPS (twice as long as train max steps to allow longer test runs)
MAX_DISTANCE_TO_GOAL = 50 # 50m
SUCCESS_DISTANCE_TO_GOAL = 0.2 # 20cm
SENSOR_WAIT_S = 0.003 # 3ms

class TestSession(Session):
    def __init__(self, config, client):
        super().__init__(config, client)
        self._mode = SessionEnum.TEST
        

    def run(self):
        print("Executing test session logic")
        
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
            
        # exit sync mode
        exit_sync_mode(world)
        
        # save model weights after training
        self._model.save_weights()
        
        # destroy ego vehicle
        self.ego_wrapper.destroy()
        
        
    def callback(self, scenario: ScenarioBase):
        print(f"[Scenario] starting at junction id {scenario._junction_id}")
        scenario.is_running = True
        
        # initialize machine learning components
        reward_function = WaypointRewardFunction(scenario._route)
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
            
            # apply the control to the ego vehicle
            vehicle.apply_control(vehicle_control)
            
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
