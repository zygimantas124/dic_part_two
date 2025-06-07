# Returns a list of obstacle positions and sizes.
# Each obstacle is represented as a tuple of (x, y, width, height).

"""
Obstacle configurations for different environments.
"""

def get_carpets(config_name="default"):
    """Get carpet obstacles for a specific configuration."""
    configs = {
        "default": [(200, 150, 80, 180)],
        "none": [],
        "many": [
            (100, 100, 60, 60),
            (300, 200, 100, 100),
            (500, 400, 80, 80),
        ],
    }
    
    return configs.get(config_name, configs["default"])


def get_people(config_name="default"):
    """Get people obstacles for a specific configuration."""
    configs = {
        "default": [(650, 100, 15)],
        "none": [],
        "crowded": [
            (150, 150, 15),
            (300, 300, 15),
            (450, 200, 15),
        ],
    }
    
    return configs.get(config_name, configs["default"])


def get_furniture(config_name="default"):
    """Get furniture obstacles for a specific configuration."""
    configs = {
        "default": [(150, 150, 20), (650, 300, 20)],
        "none": [],
        "office": [
            (150, 150, 20), 
            (300, 100, 25),  
            (500, 200, 15),  
            (650, 300, 20),  
            (400, 400, 30),  
        ],
    }
    
    return configs.get(config_name, configs["default"])