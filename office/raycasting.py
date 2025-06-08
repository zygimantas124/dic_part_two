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
    vx, vy = x2 - x1, y2 - y1
    denom = dx * vy - dy * vx
    if abs(denom) < EPSILON:
        cross1 = (x1 - px) * dy - (y1 - py) * dx
        if abs(cross1) < EPSILON:
            # Collinear: check endpoints along ray
            t1 = t2 = 0.0
            if abs(dx) > EPSILON:
                t1 = (x1 - px) / dx
                t2 = (x2 - px) / dx
            elif abs(dy) > EPSILON:
                t1 = (y1 - py) / dy
                t2 = (y2 - py) / dy
            else:
                return None
            if t1 > t2:
                t1, t2 = t2, t1
            if t2 >= -EPSILON:
                t = max(0.0, t1)
                ix = px + t*dx
                iy = py + t*dy
                return (t, (ix, iy))
        return None

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
    ox, oy = px - cx, py - cy
    a = dx*dx + dy*dy
    if a < EPSILON:
        return None
    b = 2*(dx*ox + dy*oy)
    c = ox*ox + oy*oy - r*r
    disc = b*b - 4*a*c
    if disc < -EPSILON:
        return None
    if disc < EPSILON:
        t = -b/(2*a)
        if t >= -EPSILON:
            t_clamped = max(0.0, t)
            ix = px + t_clamped*dx
            iy = py + t_clamped*dy
            return (t_clamped, (ix, iy))
        return None
    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc)/(2*a)
    t2 = (-b + sqrt_disc)/(2*a)
    valid_ts = []
    if t1 >= -EPSILON:
        valid_ts.append(max(0.0, t1))
    if t2 >= -EPSILON:
        valid_ts.append(max(0.0, t2))
    if not valid_ts:
        return None
    t_min = min(valid_ts)
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
            if inter is not None:
                t_seg, (ix_seg,iy_seg) = inter
                actual_t = t_seg + RAY_OFFSET
                if 0 <= actual_t < closest_t:
                    closest_t = actual_t
                    closest_pt = (ix_seg, iy_seg)

        # circles
        for (cx, cy, cr) in circles:
            inter = _intersect_ray_circle(ray_start_x, ray_start_y, dx, dy, cx, cy, cr)
            if inter is not None:
                t_circ, (ix_circ, iy_circ) = inter
                actual_t = t_circ + RAY_OFFSET
                if 0 <= actual_t < closest_t:
                    closest_t = actual_t
                    closest_pt = (ix_circ, iy_circ)

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
    px, py = float(robot_pos[0]), float(robot_pos[1])
    angle_step = 2*math.pi/num_rays

    wall_segments = wall_rects_to_segments(wall_rects)
    circles = [(float(cx), float(cy), float(r)+robot_radius)
               for (cx, cy, r) in circle_obstacles]

    hits_pts = []
    hits_type = []

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
    px, py = float(robot_pos[0]), float(robot_pos[1])
    half_cone = cone_width/2
    angles = [
        center_angle - half_cone + i*(cone_width/(num_rays-1))
        for i in range(num_rays)
    ]

    wall_segments = wall_rects_to_segments(wall_rects)
    circles = [(float(cx), float(cy), float(r)+robot_radius)
               for (cx, cy, r) in circle_obstacles]

    hits_pts = []
    hits_type = []

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
    px, py = float(robot_pos[0]), float(robot_pos[1])
    theta = math.radians(angle_deg)
    dx = math.cos(theta)
    dy = math.sin(theta)
    ray_start_x = px + RAY_OFFSET*dx
    ray_start_y = py + RAY_OFFSET*dy

    closest_t = max_distance
    closest_pt = None
    hit_type = "none"

    wall_segments = wall_rects_to_segments(wall_rects)
    for ((x1,y1),(x2,y2)) in wall_segments:
        inter = _intersect_ray_segment(ray_start_x, ray_start_y, dx, dy, x1, y1, x2, y2)
        if inter is not None:
            t_seg, (ix_seg, iy_seg) = inter
            actual_t = t_seg + RAY_OFFSET
            if 0 <= actual_t < closest_t:
                closest_t = actual_t
                closest_pt = (ix_seg, iy_seg)
                hit_type = "wall"

    for (cx, cy, cr) in circle_obstacles:
        inter = _intersect_ray_circle(ray_start_x, ray_start_y, dx, dy, cx, cy, cr+robot_radius)
        if inter is not None:
            t_circ, (ix_circ,iy_circ) = inter
            actual_t = t_circ + RAY_OFFSET
            if 0 <= actual_t < closest_t:
                closest_t = actual_t
                closest_pt = (ix_circ, iy_circ)
                hit_type = "circle"

    if closest_pt is None:
        ix_far = px + max_distance*dx
        iy_far = py + max_distance*dy
        closest_pt = (ix_far, iy_far)
        hit_dist = max_distance
        hit_type = "none"
    else:
        hit_dist = closest_t

    return closest_pt, hit_dist, hit_type
