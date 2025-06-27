# office/raycasting.py

import math

# ─────────────────────────────────────────────────────────────────────────────
# Constants for numerical stability and ray offset
# ─────────────────────────────────────────────────────────────────────────────
EPSILON = 1e-6       # Tolerance for floating‐point comparisons
RAY_OFFSET = 0.1     # Small offset so rays do not start exactly on surfaces

# ─────────────────────────────────────────────────────────────────────────────
# Convert each rectangular wall (x, y, w, h) into four line‐segments.
# Deduplicate overlapping segments.
# ─────────────────────────────────────────────────────────────────────────────
def wall_rects_to_segments(wall_rects):
    """
    Args:
      wall_rects: List of (x, y, width, height)
    Returns:
      List of ((x1, y1), (x2, y2)) for every unique edge of each rectangle.
    """
    segments = []
    segment_set = set()

    for (wx, wy, ww, wh) in wall_rects:
        corners = [
            (wx,        wy),        # top-left
            (wx + ww,   wy),        # top-right
            (wx + ww,   wy + wh),   # bottom-right
            (wx,        wy + wh)    # bottom-left
        ]
        for i in range(4):
            p1, p2 = corners[i], corners[(i + 1) % 4]
            if p1 <= p2:
                seg = (p1, p2)
            else:
                seg = (p2, p1)
            if seg not in segment_set:
                segment_set.add(seg)
                segments.append(seg)
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Ray–segment intersection helper
# ─────────────────────────────────────────────────────────────────────────────
def _intersect_ray_segment(px, py, dx, dy, x1, y1, x2, y2):
    """
    Helper function to find the intersection of a ray starting at (px, py)
    with a line segment with endpoints (x1, y1) and (x2, y2).
    Args:
      px, py: Ray starting point.
      dx, dy: Ray direction vector (normalized).
      x1, y1: First endpoint of the segment.
      x2, y2: Second endpoint of the segment.
    """
    vx, vy = x2 - x1, y2 - y1
    denom = dx * vy - dy * vx
    if abs(denom) < EPSILON:
        cross1 = (x1 - px) * dy - (y1 - py) * dx
        if abs(cross1) < EPSILON:
            # Collinear: check endpoints along ray
            t1 = t2 = 0.0
            # Check if the ray is parallel to a segment edge, x1 == x2 or y1 == y2
            if abs(dx) > EPSILON:
                t1 = (x1 - px) / dx
                t2 = (x2 - px) / dx
            elif abs(dy) > EPSILON:
                t1 = (y1 - py) / dy
                t2 = (y2 - py) / dy
            else:
                return None
            
            # Make t1 the smaller value and t2 the larger value
            if t1 > t2:
                t1, t2 = t2, t1
            if t2 >= -EPSILON:
                t = max(0.0, t1)
                ix = px + t*dx
                iy = py + t*dy
                return (t, (ix, iy))
        return None
    
    # Not collinear: compute intersection
    t = ((x1 - px)*vy - (y1 - py)*vx) / denom
    u = ((x1 - px)*dy - (y1 - py)*dx) / denom
    if t >= -EPSILON and -EPSILON <= u <= 1+EPSILON:
        t_clamped = max(0.0, t)
        ix = px + t_clamped*dx
        iy = py + t_clamped*dy
        return (t_clamped, (ix, iy))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Ray–circle intersection helper
# ─────────────────────────────────────────────────────────────────────────────
def _intersect_ray_circle(px, py, dx, dy, cx, cy, r):
    """
    Helper function to find the intersection of a ray starting at (px, py)
    with a circle centered at (cx, cy) with radius r.
    Args:
      px, py: Ray starting point.
      dx, dy: Ray direction vector (normalized).
      cx, cy: Circle center.
      r: Circle radius.
      
    Returns:
      (t, (ix, iy)) if intersection exists, where t is the distance along the ray
      and (ix, iy) is the intersection point. Returns None if no intersection exists.
      """
    
    # Vector from ray origin to circle center
    ox, oy = px - cx, py - cy
    
    # Coefficient 'a' for quadratic equation (t²*a + t*b + c = 0)
    a = dx*dx + dy*dy

    # Check if the direction vector is valid
    if a < EPSILON:
        return None
    
    # Coefficient 'b' and 'c' for quadratic equation
    b = 2*(dx*ox + dy*oy)
    c = ox*ox + oy*oy - r*r

    # Calculate the discriminant
    disc = b*b - 4*a*c

    # If the discriminant is negative, ray misses circle fully
    if disc < -EPSILON:
        return None
    
    # If the discriminant is close to zero, one intersection point since it is tangent
    if disc < EPSILON:
        t = -b/(2*a)
        # check if intersection is in front of the ray
        if t >= -EPSILON:
            t_clamped = max(0.0, t)
            ix = px + t_clamped*dx
            iy = py + t_clamped*dy
            return (t_clamped, (ix, iy))
        return None
    
    # If positive discriminant, two intersctions with circle
    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc)/(2*a) # Close intersection
    t2 = (-b + sqrt_disc)/(2*a) # Further intersection
    valid_ts = []
    # Collect intersections that are in front of the ray
    if t1 >= -EPSILON:
        valid_ts.append(max(0.0, t1))
    if t2 >= -EPSILON:
        valid_ts.append(max(0.0, t2))
    if not valid_ts:
        return None
    
    # Get closest intersection point
    t_min = min(valid_ts)
    # Point coors
    ix = px + t_min*dx
    iy = py + t_min*dy
    return (t_min, (ix, iy))


# ─────────────────────────────────────────────────────────────────────────────
# Ray–rectangle intersection helper
# ─────────────────────────────────────────────────────────────────────────────
def _intersect_ray_rect(px, py, dx, dy, x, y, w, h):
    hits = []
    # top
    top = _intersect_ray_segment(px, py, dx, dy, x, y, x+w, y)
    if top is not None:
        hits.append(top[0])
    # bottom
    bot = _intersect_ray_segment(px, py, dx, dy, x, y+h, x+w, y+h)
    if bot is not None:
        hits.append(bot[0])
    # left
    left = _intersect_ray_segment(px, py, dx, dy, x, y, x, y+h)
    if left is not None:
        hits.append(left[0])
    # right
    right = _intersect_ray_segment(px, py, dx, dy, x+w, y, x+w, y+h)
    if right is not None:
        hits.append(right[0])
    if not hits:
        return None
    t_min = min([t for t in hits if t >= -EPSILON] or [])
    return max(0.0, t_min)


# ─────────────────────────────────────────────────────────────────────────────
# cast_rays: shoot num_rays in 360°, return list of (ix,iy).
# ─────────────────────────────────────────────────────────────────────────────
def cast_rays(
    robot_pos,
    wall_rects,
    circle_obstacles,
    *,
    num_rays=360,
    max_distance=1000,
    robot_radius=0
):
    """
    Cast rays from the robot position in all directions.'
    Args:
      robot_pos: (x, y) position of the robot.
      wall_rects: List of (x, y, width, height) tuples for walls.
      circle_obstacles: List of (cx, cy, radius) tuples for circular obstacles.
      num_rays: Number of rays to cast in 360°.
      max_distance: Maximum distance each ray can travel.
      robot_radius: Radius of the robot for collision detection.

    Returns:
      hits (list of ix, iy) tuples where each tuple is the intersection point of a ray.
    """
    px, py = float(robot_pos[0]), float(robot_pos[1])
    angle_step = 2*math.pi/num_rays

    wall_segments = wall_rects_to_segments(wall_rects)
    circles = [(float(cx), float(cy), float(r)+robot_radius)
               for (cx, cy, r) in circle_obstacles]

    hits = []
    for i in range(num_rays):
        theta = i*angle_step
        dx = math.cos(theta)
        dy = math.sin(theta)
        ray_start_x = px + RAY_OFFSET*dx
        ray_start_y = py + RAY_OFFSET*dy

        closest_t = max_distance
        closest_pt = None

        # walls
        for ((x1,y1),(x2,y2)) in wall_segments:
            inter = _intersect_ray_segment(ray_start_x, ray_start_y, dx, dy, x1, y1, x2, y2)
            # Check if intersection exists

            if inter is not None:
                t_seg, (ix_seg,iy_seg) = inter
                actual_t = t_seg + RAY_OFFSET
                # Check if intersection in front of ray
                if 0 <= actual_t < closest_t:
                    closest_t = actual_t
                    closest_pt = (ix_seg, iy_seg)

        # circles
        for (cx, cy, cr) in circles:
            inter = _intersect_ray_circle(ray_start_x, ray_start_y, dx, dy, cx, cy, cr)
            if inter is not None:
                t_circ, (ix_circ, iy_circ) = inter
                actual_t = t_circ + RAY_OFFSET
                # Check if intersection in front of ray
                if 0 <= actual_t < closest_t:
                    closest_t = actual_t
                    closest_pt = (ix_circ, iy_circ)

        # No intersection found, return far point
        if closest_pt is None:
            ix_far = px + max_distance*dx
            iy_far = py + max_distance*dy
            hits.append((ix_far,iy_far))
        else:
            hits.append(closest_pt)

    return hits


# ─────────────────────────────────────────────────────────────────────────────
# cast_rays_with_hits: same as cast_rays, but also returns a “hit_type” per ray.
# ─────────────────────────────────────────────────────────────────────────────
def cast_rays_with_hits(
    robot_pos,
    wall_rects,
    circle_obstacles,
    *,
    num_rays=360,
    max_distance=1000,
    robot_radius=0
):
    """
    Cast rays from the robot position, also return type of hit (wall, or with obstacle).
    Args:
      robot_pos: (x, y) position of the robot.
      wall_rects: List of (x, y, width, height) tuples for walls.
      circle_obstacles: List of (cx, cy, radius) tuples for circular obstacles.
      num_rays: Number of rays to cast in 360°.
      max_distance: Maximum distance each ray can travel.
      robot_radius: Radius of the robot for collision detection.

    Returns:
      hits_pts: List of (ix, iy) tuples where each tuple is the intersection point of a ray.
      hits_type: List of hit types corresponding to each ray, either "wall", "circle", or "none".
    """

    # Get robot position
    px, py = float(robot_pos[0]), float(robot_pos[1])
    angle_step = 2*math.pi/num_rays

    # Get wall segments and circles
    wall_segments = wall_rects_to_segments(wall_rects)
    circles = [(float(cx), float(cy), float(r)+robot_radius)
               for (cx, cy, r) in circle_obstacles]

    hits_pts = []
    hits_type = []

    # Cast ray in all directions, and check for intersections
    for i in range(num_rays):
        theta = i*angle_step
        dx = math.cos(theta)
        dy = math.sin(theta)
        ray_start_x = px + RAY_OFFSET*dx
        ray_start_y = py + RAY_OFFSET*dy

        closest_t = max_distance
        closest_pt = None
        closest_type = "none"

        # walls
        for ((x1,y1),(x2,y2)) in wall_segments:
            inter = _intersect_ray_segment(ray_start_x, ray_start_y, dx, dy, x1, y1, x2, y2)
            if inter is not None:
                t_seg, (ix_seg,iy_seg) = inter
                actual_t = t_seg + RAY_OFFSET
                # Check if intersection in front of ray
                if 0 <= actual_t < closest_t:
                    closest_t = actual_t
                    closest_pt = (ix_seg,iy_seg)
                    closest_type = "wall"

        # circles
        for (cx, cy, cr) in circles:
            inter = _intersect_ray_circle(ray_start_x, ray_start_y, dx, dy, cx, cy, cr)
            if inter is not None:
                t_circ, (ix_circ, iy_circ) = inter
                actual_t = t_circ + RAY_OFFSET
                # Check if intersection in front of ray
                if 0 <= actual_t < closest_t:
                    closest_t = actual_t
                    closest_pt = (ix_circ, iy_circ)
                    closest_type = "circle"

        # No intersection found, return far point
        if closest_pt is None:
            ix_far = px + max_distance*dx
            iy_far = py + max_distance*dy
            hits_pts.append((ix_far,iy_far))
            hits_type.append("none")
        else:
            hits_pts.append(closest_pt)
            hits_type.append(closest_type)

    return hits_pts, hits_type


# ─────────────────────────────────────────────────────────────────────────────
# cast_cone_rays: cast only the rays in a 45° cone in front of the robot.
# ─────────────────────────────────────────────────────────────────────────────
def cast_cone_rays(
    robot_pos,
    wall_rects,
    circle_obstacles,
    *,
    center_angle,
    cone_width,
    num_rays=30,
    max_distance=1000,
    robot_radius=0
):
    """
    Cast rays in a cone shape in front of the robot.
    Args:
      robot_pos: (x, y) position of the robot.
      wall_rects: List of (x, y, width, height) tuples for walls.
      circle_obstacles: List of (cx, cy, radius) tuples for circular obstacles. 
      center_angle: Center angle of the cone in radians.
      cone_width: Width of the cone in radians.
      num_rays: Number of rays to cast in the cone.
      max_distance: Maximum distance each ray can travel.
      robot_radius: Radius of the robot for collision detection.
    Returns:
        hits_pts: List of (ix, iy) tuples where each tuple is the intersection point of a ray.  
        hits_type: List of hit types corresponding to each ray, either "wall", "circle", or "none".
    """

    # Get robot position and calculate angles for rays
    px, py = float(robot_pos[0]), float(robot_pos[1])
    half_cone = cone_width/2
    # Get cone angles
    angles = [
        center_angle - half_cone + i*(cone_width/(num_rays-1))
        for i in range(num_rays)
    ]

    
    # Get wall segments and circles
    wall_segments = wall_rects_to_segments(wall_rects)
    circles = [(float(cx), float(cy), float(r)+robot_radius)
               for (cx, cy, r) in circle_obstacles]

    hits_pts = []
    hits_type = []

    # Cast ray in cone direction, and check for intersections
    for theta in angles:
        dx = math.cos(theta)
        dy = math.sin(theta)
        ray_start_x = px + RAY_OFFSET*dx
        ray_start_y = py + RAY_OFFSET*dy

        closest_t = max_distance
        closest_pt = None
        closest_type = "none"

        # walls
        for ((x1,y1),(x2,y2)) in wall_segments:
            inter = _intersect_ray_segment(ray_start_x, ray_start_y, dx, dy, x1, y1, x2, y2)
            if inter is not None:
                t_seg, (ix_seg, iy_seg) = inter
                actual_t = t_seg + RAY_OFFSET
                if 0 <= actual_t < closest_t:
                    closest_t = actual_t
                    closest_pt = (ix_seg, iy_seg)
                    closest_type = "wall"

        # circles
        for (cx, cy, cr) in circles:
            inter = _intersect_ray_circle(ray_start_x, ray_start_y, dx, dy, cx, cy, cr)
            if inter is not None:
                t_circ, (ix_circ,iy_circ) = inter
                actual_t = t_circ + RAY_OFFSET
                if 0 <= actual_t < closest_t:
                    closest_t = actual_t
                    closest_pt = (ix_circ, iy_circ)
                    closest_type = "circle"

        if closest_pt is None:
            ix_far = px + max_distance*dx
            iy_far = py + max_distance*dy
            hits_pts.append((ix_far,iy_far))
            hits_type.append("none")
        else:
            hits_pts.append(closest_pt)
            hits_type.append(closest_type)

    return hits_pts, hits_type


# ─────────────────────────────────────────────────────────────────────────────
# debug_cast_single_ray: for angle_deg (degrees), return (hit_pt, hit_dist, hit_type)
# ─────────────────────────────────────────────────────────────────────────────
def debug_cast_single_ray(
    robot_pos,
    angle_deg,
    wall_rects,
    circle_obstacles,
    *,
    max_distance=1000,
    robot_radius=0
):
    """
    Cast a single ray from the robot position at a specific angle.
    Args:
        robot_pos: (x, y) position of the robot.
        angle_deg: Angle in degrees at which to cast the ray.
        wall_rects: List of (x, y, width, height) tuples for walls.
        circle_obstacles: List of (cx, cy, radius) tuples for circular obstacles.
        max_distance: Maximum distance the ray can travel.
        robot_radius: Radius of the robot for collision detection.
    Returns:
      (closest_pt, hit_dist, hit_type):
        closest_pt: (ix, iy) tuple of the intersection point.
        hit_dist: Distance from the robot to the intersection point.
        hit_type: Type of hit ("wall", "circle", or "none").
    """
    # Get robot position and calculate ray direction
    px, py = float(robot_pos[0]), float(robot_pos[1])
    theta = math.radians(angle_deg)
    dx = math.cos(theta)
    dy = math.sin(theta)
    ray_start_x = px + RAY_OFFSET*dx
    ray_start_y = py + RAY_OFFSET*dy

    # Initialize closest point and distance
    closest_t = max_distance
    closest_pt = None
    hit_type = "none"

    # Convert wall rectangles to segments
    wall_segments = wall_rects_to_segments(wall_rects)

    # Check for all wall segments if a collission occurs
    for ((x1,y1),(x2,y2)) in wall_segments:
        inter = _intersect_ray_segment(ray_start_x, ray_start_y, dx, dy, x1, y1, x2, y2)
        if inter is not None:
            t_seg, (ix_seg, iy_seg) = inter
            actual_t = t_seg + RAY_OFFSET
            # Check if intersection in front of ray
            if 0 <= actual_t < closest_t:
                closest_t = actual_t
                closest_pt = (ix_seg, iy_seg)
                hit_type = "wall"

    # Check for all circles if a collission occurs
    for (cx, cy, cr) in circle_obstacles:
        inter = _intersect_ray_circle(ray_start_x, ray_start_y, dx, dy, cx, cy, cr+robot_radius)
        if inter is not None:
            t_circ, (ix_circ,iy_circ) = inter
            actual_t = t_circ + RAY_OFFSET
            # Check if intersection in front of ray
            if 0 <= actual_t < closest_t:
                closest_t = actual_t
                closest_pt = (ix_circ, iy_circ)
                hit_type = "circle"

    # If no intersection found, return far point
    if closest_pt is None:
        ix_far = px + max_distance*dx
        iy_far = py + max_distance*dy
        closest_pt = (ix_far, iy_far)
        hit_dist = max_distance
        hit_type = "none"
    else:
        hit_dist = closest_t

    return closest_pt, hit_dist, hit_type
