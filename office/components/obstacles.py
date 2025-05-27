# Returns a list of obstacle positions and sizes.
# Each obstacle is represented as a tuple of (x, y, width, height).


def get_carpets():
    return [(200, 150, 80, 180)]


def get_people():
    return [
        (200, 100, 15),
        (700, 100, 15),
    ]


def get_furniture():
    return [
        (200, 150, 20),
        (700, 150, 20),
    ]
