import yaml

def load_config(path) -> dict:
    """
    Loads and returns configuration dict from a yaml file.
    Note: Assumes proper structure and keys are present.
    """
    
    with open(path, "r") as file:
        config = yaml.safe_load(file)
        
    print(f"Configuration loaded from {path}")
    return config

def write_summary(summary: dict, path) -> None:
    """
    Writes the session summary to a file.
    """
    
    with open(path, "w") as file:
        yaml.dump(summary, file)
    print(f"Summary written to {path}")
