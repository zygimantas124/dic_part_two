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
        furniture="complex",
    ),
    
    "open_office": EnvironmentConfig(
        name="open_office",
        walls="open_office",
        tables="open_office",
        carpets="none",
        people="open_office",
        furniture="open_office",
    ),

    "open_office_simple": EnvironmentConfig(
        name="open_office_simple",
        walls="open_office_simple",
        tables="open_office",
        carpets="none",
        people="none",
        furniture="none",
    ),

    "big_table": EnvironmentConfig(
        name="big_table",
        walls="simple",
        tables="big_table",
        carpets="none",
        people="none",
        furniture="none",
    ),
}

def get_config(name):
    """Get a pre-defined environment configuration."""
    if name not in CONFIGS:
        raise ValueError(f"Unknown configuration: {name}")
    return CONFIGS[name]