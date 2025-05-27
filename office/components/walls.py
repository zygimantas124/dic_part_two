# Returns a list of wall positions and sizes.
# Each wall is represented as a tuple of (x, y, width, height).


def get_border_walls():
    return [
        (0, 0, 800, 6),  # Top
        (0, 0, 8, 600),  # Left
        (792, 0, 8, 600),  # Right
        (0, 594, 800, 6),  # Bottom
    ]


def get_horizontal_interior_walls():
    return [
        (0, 60, 80, 6),  # (0, 1)
        (160, 60, 160, 6),  # (2, 1)
        (400, 60, 400, 6),  # (5, 1)
        (400, 360, 400, 6),  # (5, 6)
        (0, 420, 320, 6),  # (0, 7)
        (400, 420, 400, 6),  # (5, 7)
    ]


def get_vertical_interior_walls():
    return [
        (320, 60, 6, 180),  # (4,1) height=3
        (320, 300, 6, 120),  # (4,5) height=2
        (320, 480, 6, 120),  # (4,8) height=2
        (400, 60, 6, 60),  # (5,1)
        (400, 180, 6, 180),  # (5,3)
        (400, 480, 6, 120),  # (5,8)
    ]


def get_walls():
    return get_border_walls() + get_horizontal_interior_walls() + get_vertical_interior_walls()
