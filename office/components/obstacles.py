# Returns a list of obstacle positions and sizes.
# Each obstacle is represented as a tuple of (x, y, width, height).

"""
Obstacle configurations for different environments.
"""

def get_carpets(config_name="none"):
    """Get carpet obstacles for a specific configuration."""
    configs = {
        "complex": [(200, 150, 80, 180)],
        "none": [],
    }
    
    return configs.get(config_name, configs["none"])


def get_people(config_name="none"):
    """Get people obstacles for a specific configuration."""
    configs = {
        "open_office": [(700, 300, 15)],
        "open_office_simple": [(600, 200, 15)],
        "none": [],
        "complex": [(650, 100, 15)],
    }
    
    return configs.get(config_name, configs["none"])


def get_furniture(config_name="none"):
    """Get furniture obstacles for a specific configuration."""
    configs = {
        "open_office": [(150, 150, 40), 
                    (600, 475, 30)],
        "open_office_simple": [(150, 250, 40), 
                            (500, 400, 30)],
        "none": [],
        "complex": [
        (150, 150, 20),
        (650, 300, 20)],
    }
    
    return configs.get(config_name, configs["none"])