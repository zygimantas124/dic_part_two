# Returns a list of target table positions and sizes.
# Each table is represented as a tuple of (x, y, width, height).

def get_target_tables(scale=0.7):  
    original_tables = [
        (10, 120, 80, 120),
        (80, 360, 160, 60),
        (405, 240, 80, 120),
        (710, 120, 80, 120),
        (480, 530, 240, 60),
    ]
    
    scaled_tables = []
    for x, y, width, height in original_tables:
        # current center
        center_x = x + width / 2
        center_y = y + height / 2
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # calc new position to keep center the same
        new_x = int(center_x - new_width / 2)
        new_y = int(center_y - new_height / 2)
        
        scaled_tables.append((new_x, new_y, new_width, new_height))
    
    return scaled_tables