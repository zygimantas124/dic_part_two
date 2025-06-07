"""
Pre-defined environment configurations combining walls, tables, and obstacles.
"""

class EnvironmentConfig:
    """Container for a complete environment configuration."""
    def __init__(self, name, walls="simple", tables="simple", carpets="simple", 
                 people="simple", furniture="simple", table_scale=0.7,
                 start_pos=None, table_priorities=None):
        self.name = name
        self.walls = walls
        self.tables = tables
        self.carpets = carpets
        self.people = people
        self.furniture = furniture
        self.table_scale = table_scale
        self.start_pos = start_pos
        self.table_priorities = table_priorities

# Pre-defined configurations
CONFIGS = {
        
    "simple": EnvironmentConfig(
        name="simple",
        walls="simple",
        tables="simple",
        carpets="none",
        people="none",
        furniture="none",
    ),
    
    "complex": EnvironmentConfig(
        name="complex",
        walls="complex",
        tables="complex",
        carpets="complex",
        people="crowded",
        furniture="office",
    ),
    
    "open_office": EnvironmentConfig(
        name="open_office",
        walls="open_office",
        tables="open_office",
        carpets="none",
        people="open_office",
        furniture="open_office",
    ),
}

def get_config(name):
    """Get a pre-defined environment configuration."""
    if name not in CONFIGS:
        raise ValueError(f"Unknown configuration: {name}")
    return CONFIGS[name]