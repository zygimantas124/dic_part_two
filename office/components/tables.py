# Returns a list of target table positions and sizes.
# Each table is represented as a tuple of (x, y, width, height).

"""
Table configurations for different environments.
Each configuration returns a list of tables as (x, y, width, height) tuples.
"""

def get_target_tables(config_name="simple", scale=0.8):
    """
    Get tables for a specific configuration.
    
    Args:
        config_name (str): Name of the configuration to load
        scale (float): Scale factor for table sizes
        
    Returns:
        list: List of table tuples (x, y, width, height)
    """
    configs = {
        "complex": _get_complex_tables,
        "simple": _get_simple_tables,
        "open_office": _get_open_office_tables,
        "big_table": _get_big_table,
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown table configuration: {config_name}")
    
    tables = configs[config_name]()
    
    # Apply scaling if needed
    if scale != 1.0:
        tables = _scale_tables(tables, scale)
    
    return tables


def _scale_tables(tables, scale):
    """Scale tables while maintaining their centers."""
    scaled_tables = []
    for x, y, width, height in tables:
        center_x = x + width / 2
        center_y = y + height / 2
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        new_x = int(center_x - new_width / 2)
        new_y = int(center_y - new_height / 2)
        
        scaled_tables.append((new_x, new_y, new_width, new_height))
    
    return scaled_tables



####################### TABLE DEFINITIONS #######################

########## For COMPLEX TABLES ##########
def _get_complex_tables():
    """Complex table configuration (original)."""
    return [
        (10, 120, 80, 120),
        (80, 360, 160, 60),
        (405, 240, 80, 120),
        (710, 120, 80, 120),
        (480, 530, 240, 60),
    ]

########## For SIMPLE TABLES ##########
def _get_simple_tables():
    """Simple configuration with just 1 table."""
    return [
        (600, 50, 150, 100),
    ]

########## For OPEN OFFICE TABLES ##########
def _get_open_office_tables():
    """Simple configuration with just 1 table."""
    return [
        (600, 50, 120, 80),
    ]

########## For BIG TABLE ##########
def _get_big_table():
    """Big table configuration."""
    return [
        (100, 0, 600, 600),
    ]