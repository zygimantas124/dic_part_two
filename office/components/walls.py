# Returns a list of wall positions and sizes.
# Each wall is represented as a tuple of (x, y, width, height).


def get_walls(config_name="simple"):
    """
    Get walls for a specific configuration.
    
    Args:
        config_name (str): Name of the configuration to load
        
    Returns:
        list: List of wall tuples (x, y, width, height)
    """
    configs = {
        "simple": _get_simple_walls,
        "complex": _get_complex_walls,
        "open_office": _get_open_office_walls,
        "open_office_simple": _get_open_office_simple_walls,
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown wall configuration: {config_name}")
    
    return configs[config_name]()

def _get_complex_walls():
    """Complex configuration (starting of assignment)."""
    return (
        _get_border_walls() + 
        _get_horizontal_interior_walls() + 
        _get_vertical_interior_walls()
    )

def _get_simple_walls():
    """Simple environment with just borders."""
    return _get_border_walls()


def _get_open_office_walls():
    """Open office layout with minimal walls."""
    return (
        _get_border_walls() +
        _get_office_horizontal_walls() +
        _get_office_vertical_walls()
    )

def _get_open_office_simple_walls():
    """Open office Simple layout with minimal walls."""
    return (
        _get_border_walls() +
        _get_office_simple_horizontal_walls() +
        _get_office_simple_vertical_walls()
    )


####################### WALL DEFINITIONS #######################

def _get_border_walls():
    return [
        (0, 0, 800, 6),      # Top
        (0, 0, 8, 600),      # Left
        (792, 0, 8, 600),    # Right
        (0, 594, 800, 6),    # Bottom
    ]

########## For COMPLEX WALLS ##########
def _get_horizontal_interior_walls():
    return [
        (0, 60, 80, 6),
        (160, 60, 160, 6),
        (400, 60, 400, 6),
        (400, 360, 400, 6),
        (0, 420, 320, 6),
        (400, 420, 400, 6),
    ]

def _get_vertical_interior_walls():
    return [
        (320, 60, 6, 180),
        (320, 300, 6, 120),
        (320, 480, 6, 120),
        (400, 60, 6, 60),
        (400, 180, 6, 180),
        (400, 480, 6, 120),
    ]

########## For OPEN OFFICE WALLS ##########

def _get_office_horizontal_walls():
    return [    ]
def _get_office_vertical_walls():
    return [
        (250, 200, 6, 400),    
        (450, 0, 6, 450),      
    ]


########## For OPEN OFFICE SIMPLE WALLS ##########

def _get_office_simple_horizontal_walls():
    return [    ]
def _get_office_simple_vertical_walls():
    return [
        (250, 400, 20, 250),    
        (450, 0, 20, 250),      
    ]


