import signal
from utils import misc
from session.factory import SessionFactory
from simulator.interface import *
import argparse

CONFIG_PATH = "config.yml"


def signal_handler(sig, frame):
    print(f"Process received signal {sig}.")
    sys.exit(0)


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGSEGV, signal_handler)


def main():
    """
    Main activation
    """
    # add argument parsing
    parser = argparse.ArgumentParser(description="CARLA Simulation Script")
    parser.add_argument(
        "-j",
        "--show-junctions",
        action="store_true",
        help="Show all junctions in the map and exit",
    )
    parser.add_argument(
        "-t", "--run-tests", action="store_true", help="Run tests and exit"
    )
    args = parser.parse_args()

    # get client and config
    client, config = load_client_and_config(CONFIG_PATH)

    # reload world to ensure a clean state
    client.reload_world()

    # argument-based operations
    # display junctions
    if args.show_junctions:
        print("Debugging junctions...")
        debug_junctions(client)
        exit(0)

    # run tests
    elif args.run_tests:
        # [todo] implement test suite
        print("Test suite not implemented yet.")
        exit(0)

    # normal operation
    # initialize and run session
    try:
        # pre-clean debug residues
        session = SessionFactory.create_session(config["runtime"], client=client)
        session.run()
    except Exception as e:
        print(f"[Error] Runtime failed: {e}")
        print(session.summary)
        exit(1)

    # for a successful run print the summary
    print(session.summary)


def load_client_and_config(config_path):
    # load runtime configuration
    config = misc.load_config(config_path)

    # load server configuration
    host = config["server"].get("host", "localhost")
    port = config["server"].get("port", 2000)

    # load carla client
    return connect_to_carla_server(host, port), config


def debug_junctions(client):
    # [debug] visualize all junction routes
    routes = get_junction_routes(client.get_world().get_map())
    draw_and_spectate_junction_routes_iteratively(client, routes, delay=1.0)


if __name__ == "__main__":
    main()
