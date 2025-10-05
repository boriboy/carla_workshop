from .scenario_base import ScenarioBase

class EmptyScenario(ScenarioBase):
    def __init__(self, config, client, map, ego_wrapper, junction_id, route):
        # init ego position, route, spectator positioning, etc.
        super().__init__(config, client, map, ego_wrapper, junction_id, route)
        
        # scenario is empty therefore no additional actors are spawned
        print("[EmptyScenario] initialized")