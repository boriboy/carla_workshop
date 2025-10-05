"""
Utility functions for CARLA client and world objects.
"""

import math
import sys
import glob
import os
import time
from random import choice
import numpy as np


# load carla module
try:
    sys.path.append(
        glob.glob(
            "../PythonAPI/carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass

import carla


# connect to CARLA server
def connect_to_carla_server(host="localhost", port=2000):
    """Connect to the CARLA server."""
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    return client


# enter sync mode
def enter_sync_mode(world, fixed_delta_seconds=0.05):
    """Set the world to synchronous mode."""
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta_seconds
    world.apply_settings(settings)
    print("[World] set to synchronous mode.")


def exit_sync_mode(world):
    """Set the world back to asynchronous mode."""
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    print("[World] set to asynchronous mode.")


# get all junctions in the map
def get_all_junctions(carla_map):
    """Retrieve all junctions in the CARLA map."""
    waypoints = carla_map.generate_waypoints(distance=5.0)
    junctions = dict()

    for wp in waypoints:
        if wp.is_junction:
            jnct = wp.get_junction()
            junctions[jnct.id] = jnct

    return list(junctions.values())


def get_junction_routes(carla_map):
    """Get all possible routes through all junctions in the map."""
    junctions = get_all_junctions(carla_map)
    routes = dict()

    for junction in junctions:
        routes[junction.id] = collect_routes_in_junction(junction)

    return routes


def draw_and_spectate_junction_routes_iteratively(client, junction_routes, delay=2.0):
    """Iterate through junction routes, drawing and spectating each."""
    for junction_id, routes in junction_routes.items():
        for route_wps in routes:
            entry_wp = route_wps[0]
            exit_wp = route_wps[-1]

            # place spectator view at middle of the route and draw the route
            place_spectator_on_transform(
                client.get_world(), route_wps[len(route_wps) // 2].transform
            )
            draw_waypoints(client.get_world(), entry_wp.next_until_lane_end(0.1), delay)
            draw_waypoints(client.get_world(), [exit_wp], delay, highlight=True)
            time.sleep(delay)


def get_vehicle_control_object(throttle, steer, brake):
    """Create and return a carla.VehicleControl object."""
    return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)


def actions_to_control(action_mean):
    # extract and interpolate action values
    throttle, steer, brake = action_mean.flatten().tolist()

    # convert to applicable ranges
    control = carla.VehicleControl()
    control.throttle = np.clip((throttle + 1) / 2, 0.0, 1.0)  # [0, 1]
    control.steer = np.clip(steer, -1.0, 1.0)  # [-1, 1]
    control.brake = np.clip((brake + 1) / 2, 0.0, 1.0)  # [0, 1]

    # conditional logic to avoid simultaneous throttle and brake
    if throttle > brake:
        control.brake = 0.0
    else:
        control.throttle = 0.0

    return control


import carla
import random
import math


def spawn_pedestrians_around_wp(
    client,
    reference_location: carla.Location,
    num_pedestrians=5,
    radius=15.0,
    min_distance=3.0,
):
    """
    Spawn pedestrians around a reference location within a given radius.

    Args:
        client: CARLA client instance
        reference_location: carla.Location - Center point for spawning
        num_pedestrians: int - Number of pedestrians to spawn
        radius: float - Maximum distance from reference location (in meters)
        min_distance: float - Minimum distance between pedestrians (in meters)

    Returns:
        list: List of spawned pedestrian actor IDs [walker_id, controller_id, ...]
    """

    if num_pedestrians == 0:
        return []

    world = client.get_world()
    carla_map = world.get_map()

    # Get pedestrian blueprints (filter out children for safety)
    blueprints = world.get_blueprint_library().filter("walker.pedestrian.*")

    # Find valid spawn locations within radius
    valid_spawn_points = []
    spawn_attempts = 0
    max_attempts = num_pedestrians * 20  # Try more locations than needed

    while len(valid_spawn_points) < num_pedestrians and spawn_attempts < max_attempts:
        spawn_attempts += 1

        # Generate random position within radius
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(min_distance, radius)

        offset_x = distance * math.cos(angle)
        offset_y = distance * math.sin(angle)

        spawn_location = carla.Location(
            x=reference_location.x + offset_x,
            y=reference_location.y + offset_y,
            z=reference_location.z + 1.0,  # Slight elevation
        )

        # Get nearest waypoint (sidewalk preferred)
        waypoint = carla_map.get_waypoint(
            spawn_location,
            project_to_road=True,
            lane_type=(carla.LaneType.Sidewalk | carla.LaneType.Shoulder),
        )

        if waypoint is None:
            # Try driving lane as fallback
            waypoint = carla_map.get_waypoint(
                spawn_location, project_to_road=True, lane_type=carla.LaneType.Driving
            )

        if waypoint is None:
            continue

        spawn_transform = waypoint.transform
        spawn_transform.location.z += 0.2  # Small elevation to avoid ground collision

        # Check minimum distance from already selected spawn points
        too_close = False
        for existing_spawn in valid_spawn_points:
            dist = spawn_transform.location.distance(existing_spawn["location"])
            if dist < min_distance:
                too_close = True
                break

        if not too_close:
            valid_spawn_points.append(
                {
                    "transform": spawn_transform,
                    "location": spawn_transform.location,
                    "distance_from_reference": reference_location.distance(
                        spawn_transform.location
                    ),
                }
            )

    if not valid_spawn_points:
        print("No valid pedestrian spawn locations found within radius!")
        return []

    # Limit to requested number of pedestrians
    num_pedestrians = min(num_pedestrians, len(valid_spawn_points))
    selected_spawns = valid_spawn_points[:num_pedestrians]

    print(f"Found {len(selected_spawns)} valid spawn points for pedestrians")

    # Spawn pedestrians using batch commands
    SpawnActor = carla.command.SpawnActor

    # Walker controller blueprint
    # walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    pedestrian_batch = []

    for spawn_data in selected_spawns:
        blueprint = random.choice(blueprints)

        # Randomize pedestrian appearance
        if blueprint.has_attribute("is_invincible"):
            blueprint.set_attribute("is_invincible", "false")

        pedestrian_batch.append(
            SpawnActor(blueprint, spawn_data["transform"])
            # .then(SpawnActor(walker_controller_bp, spawn_data['transform'],))
        )

    # Execute batch spawn
    pedestrian_results = client.apply_batch_sync(pedestrian_batch, True)

    # Collect spawned pedestrian IDs
    pedestrian_actors = []
    spawned_pedestrians = []

    for i, response in enumerate(pedestrian_results):
        if not response.error:
            pedestrian_actors.append(response.actor_id)
            distance = selected_spawns[i]["distance_from_reference"]
            # print(f"Spawned pedestrian {response.actor_id} ({distance:.1f}m from reference)")
            spawned_pedestrians.append(world.get_actor(response.actor_id))

    # Now spawn walker AI controllers for each pedestrian
    walker_controller_bp = world.get_blueprint_library().find("controller.ai.walker")

    controller_batch = []
    for pedestrian in spawned_pedestrians:
        controller_batch.append(
            SpawnActor(walker_controller_bp, pedestrian.get_transform(), pedestrian)
        )

    # Execute controller batch spawn
    controller_results = client.apply_batch_sync(controller_batch, True)

    controller_actors = []

    for i, response in enumerate(controller_results):
        if response.error:
            print(
                f"Failed to spawn controller for pedestrian {pedestrian_actors[i]}: {response.error}"
            )
        else:
            controller_actors.append(response.actor_id)

    # Start the controllers (make pedestrians walk)

    for i, controller_id in enumerate(controller_actors):
        controller = world.get_actor(controller_id)
        if controller:
            # Start walking with random speed
            controller.start()
            loc = world.get_random_location_from_navigation()
            # below line ocasionally causes segfault in CARLA client - it's a known issue in 0.9.10
            # https://github.com/carla-simulator/carla/issues/8805
            controller.go_to_location(loc)
            controller.set_max_speed(random.uniform(1.0, 2.0))  # 1-2 m/s walking speed

    print(
        f"Successfully spawned {len(spawned_pedestrians)} pedestrians with AI controllers"
    )

    return pedestrian_actors


# Alternative version with waypoint-based spawning for more precise placement
def spawn_vehicles_around_wp(
    client,
    reference_location: carla.Location,
    ignore_traffic_lights=True,
    num_vehicles=5,
    search_distance=30.0,
    search_resolution=3.0,
):
    """Spawn vehicles using waypoint navigation around junction"""

    if num_vehicles == 0:
        return []

    world = client.get_world()
    carla_map = world.get_map()

    # Get waypoint at reference location
    reference_waypoint = carla_map.get_waypoint(
        reference_location, project_to_road=True, lane_type=carla.LaneType.Driving
    )

    # Find valid spawn locations using waypoint navigation
    valid_spawn_points = []

    # Go backwards and forwards along the road
    for direction in ["previous", "next"]:
        temp_wp = reference_waypoint
        distance = 0.0

        while distance < search_distance:
            if direction == "previous":
                next_waypoints = temp_wp.previous(search_resolution)
            else:
                next_waypoints = temp_wp.next(search_resolution)

            if not next_waypoints:
                break

            temp_wp = next_waypoints[0]
            distance += search_resolution

            # Check if it's not in a junction (safer spawn location)
            if not temp_wp.is_junction:
                # Add some offset to avoid exact road center
                spawn_transform = temp_wp.transform
                spawn_transform.location.z += 0.1  # Slight elevation

                valid_spawn_points.append(
                    {
                        "spawn_point": spawn_transform,
                        "distance_from_reference": reference_location.distance(
                            spawn_transform.location
                        ),
                        "waypoint": temp_wp,
                    }
                )

    # Also check adjacent lanes
    for lane_change in [
        reference_waypoint.get_left_lane(),
        reference_waypoint.get_right_lane(),
    ]:
        if lane_change and lane_change.lane_type == carla.LaneType.Driving:
            spawn_transform = lane_change.transform
            spawn_transform.location.z += 0.1

            valid_spawn_points.append(
                {
                    "spawn_point": spawn_transform,
                    "distance_from_reference": reference_location.distance(
                        spawn_transform.location
                    ),
                    "waypoint": lane_change,
                }
            )

    if not valid_spawn_points:
        print("No valid waypoint-based spawn locations found!")
        return []

    # Remove duplicates and sort by distance
    valid_spawn_points = list(
        {v["waypoint"].id: v for v in valid_spawn_points}.values()
    )
    valid_spawn_points.sort(key=lambda x: x["distance_from_reference"])

    # Select random spawn points
    num_vehicles = min(num_vehicles, len(valid_spawn_points))
    selected_spawns = random.sample(valid_spawn_points, num_vehicles)

    # Now spawn vehicles (same logic as before)
    return spawn_vehicles_from_transforms(
        client, selected_spawns, ignore_traffic_lights
    )


def spawn_vehicles_from_transforms(client, spawn_data_list, ignore_traffic_lights):
    """Helper function to spawn vehicles from transform data"""
    world = client.get_world()
    blueprints = world.get_blueprint_library().filter("vehicle.*")
    safe_blueprints = [
        x for x in blueprints if int(x.get_attribute("number_of_wheels")) == 4
    ]

    traffic_manager = client.get_trafficmanager()
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor

    batch = []
    vehicles_list = []

    for spawn_data in spawn_data_list:
        blueprint = choice(safe_blueprints)

        if blueprint.has_attribute("color"):
            color = choice(blueprint.get_attribute("color").recommended_values)
            blueprint.set_attribute("color", color)

        blueprint.set_attribute("role_name", "autopilot")

        batch.append(
            SpawnActor(blueprint, spawn_data["spawn_point"]).then(
                SetAutopilot(FutureActor, True, traffic_manager.get_port())
            )
        )

    responses = client.apply_batch_sync(batch, True)

    for i, response in enumerate(responses):
        if response.error:
            print(f"Failed to spawn vehicle {i}: {response.error}")
        else:
            vehicles_list.append(response.actor_id)
            set_ignore_traffic_lights(
                client, world, response.actor_id, ignore_traffic_lights
            )

    return vehicles_list


def set_ignore_traffic_lights(client, world, vehicle_id, ignore=True):
    """Set whether a vehicle should ignore traffic lights."""
    vehicle = world.get_actor(vehicle_id)
    if vehicle is not None:
        traffic_manager = client.get_trafficmanager()
        percentage = 100 if ignore else 0
        traffic_manager.ignore_lights_percentage(vehicle, percentage)


# clean up function
def cleanup_aux_actors(client, actor_ids):
    """Destroy spawned actors given their IDs"""
    try:
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_ids], True)
        print(f"Destroyed {len(actor_ids)} actors.")
    except Exception as e:
        print(f"Error during cleanup: {e}")


def place_spectator_on_transform(world, transform, height=60.0, pitch=-90.0):
    """Place the spectator camera above a specific location"""
    spectator = world.get_spectator()
    spectator_location = transform.location
    spectator_location.z += height
    rotation = transform.rotation
    rotation.pitch = pitch  # down angled view
    spectator.set_transform(
        carla.Transform(
            spectator_location, carla.Rotation(pitch=rotation.pitch, yaw=0, roll=0)
        )
    )


def spawn_ego_on_transform(world, ego_bp, transform: carla.Transform):
    """Spawn the ego vehicle on a specific transform"""
    ego_bp.set_attribute("role_name", "ego")
    transform.location.z += 0.1  # slight elevation to avoid collision with ground
    return world.spawn_actor(ego_bp, transform)


def get_blueprint_by_name(world, name):
    """Retrieve a blueprint by its name from the world"""
    blueprint = world.get_blueprint_library().find(name)
    if not blueprint:
        raise ValueError(f"Blueprint '{name}' not found in the world.")
    return blueprint


def collect_routes_in_junction(junction, distance=3):
    """Collect all possible routes through a junction"""
    junction_routes = []
    junction_waypoints = junction.get_waypoints(carla.LaneType.Driving)

    for entry_wp, exit_wp in junction_waypoints:
        route = entry_wp.next_until_lane_end(distance)
        # route.append(exit_wp)   # ensure exit waypoint is included
        junction_routes.append(route)

    return junction_routes


def draw_waypoints(world, waypoints, life_time=30.0, highlight=None):
    """Draw waypoints as points and arrows in the simulator"""
    debug = world.debug

    for waypoint in waypoints:
        # Draw waypoint as a point
        debug.draw_point(
            waypoint.transform.location + carla.Location(z=0.5),
            size=0.3 if highlight else 0.1,
            color=(
                carla.Color(0, 0, 255) if highlight else carla.Color(0, 255, 0)
            ),  # Green
            life_time=life_time,
        )

        # Draw direction arrow
        debug.draw_arrow(
            waypoint.transform.location,
            waypoint.transform.location + waypoint.transform.get_forward_vector() * 2,
            thickness=0.05,
            arrow_size=0.2,
            color=carla.Color(255, 0, 0),  # Red
            life_time=life_time,
        )
