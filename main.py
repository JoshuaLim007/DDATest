import numpy as np
from noise import pnoise2
import pygame
import math
import matplotlib.pyplot as plt

def generate_perlin_noise_2d(shape, res):
    def f(x, y):
        return pnoise2(x / res,
                       y / res,
                       octaves=6,
                       persistence=0.5,
                       lacunarity=2.0,
                       repeatx=1024,
                       repeaty=1024,
                       base=42)
    return np.vectorize(f)(*np.indices(shape))

def create_voxel_buffer(shape, threshold=0.1):
    noise = generate_perlin_noise_2d(shape, res=100)
    return noise > threshold

def create_lower_resolution_buffer(buffer, factor=4):
    low_res_shape = (buffer.shape[0] // factor, buffer.shape[1] // factor)
    low_res_buffer = np.zeros(low_res_shape, dtype=bool)
    for y in range(low_res_shape[0]):
        for x in range(low_res_shape[1]):
            low_res_buffer[y, x] = np.any(buffer[y*factor:(y+1)*factor, x*factor:(x+1)*factor])
    return low_res_buffer

def ray_intersects_aabb(ray_origin, ray_direction, box_min, box_max):
    inv_dir = 1.0 / np.array(ray_direction)
    t_min_vals = (np.array(box_min) - np.array(ray_origin)) * inv_dir
    t_max_vals = (np.array(box_max) - np.array(ray_origin)) * inv_dir

    t1 = np.minimum(t_min_vals, t_max_vals)
    t2 = np.maximum(t_min_vals, t_max_vals)

    t_min = np.max(t1)  # Largest entering time
    t_max = np.min(t2)  # Smallest exiting time

    if t_max < max(t_min, 0):  
        return False, None  # No intersection

    intersection_point = np.array(ray_origin) + t_min * np.array(ray_direction)
    return True, intersection_point

def dda_ray_traversal(grid, start, direction, bounds=None, max_steps=100, per_voxel_bounds=None, per_voxel_bounds_scale = 4):
    """
    Perform DDA ray traversal on a 2D grid.
   
    :param grid: 2D list of on (1) and off (0) values
    :param start: (x, y) start position
    :param direction: (dx, dy) direction vector (should be normalized)
    :param bounds: Optional bounds (min_x, min_y, max_x, max_y) to restrict traversal
    :param max_steps: Maximum number of steps to take
    :return: List of traversed cells and exact intersection points on voxel edges
    """
    x, y = start
    dx, dy = direction
   
    # Grid dimensions
    rows, cols = len(grid), len(grid[0])
   
    # Grid cell indices
    cell_x, cell_y = int(x), int(y)
   
    # Step directions
    step_x = 1 if dx > 0 else -1
    step_y = 1 if dy > 0 else -1
   
    # Compute tDelta (how far we travel in t before hitting next grid line)
    tDelta_x = abs(1 / dx) if dx != 0 else float('inf')
    tDelta_y = abs(1 / dy) if dy != 0 else float('inf')
   
    # Compute tMax (when we first cross a grid boundary)
    tMax_x = ((cell_x + (step_x > 0)) - x) / dx if dx != 0 else float('inf')
    tMax_y = ((cell_y + (step_y > 0)) - y) / dy if dy != 0 else float('inf')
   
    traversed = []
    intersected_points = []
    intersected_points.append((x, y))
    
    hit = False
    out_of_bounds = False
    steps_taken = 0

    for step in range(max_steps):
        if 0 <= cell_x < cols and 0 <= cell_y < rows:
            traversed.append((cell_x, cell_y))
            if per_voxel_bounds:
                perVxbounds = per_voxel_bounds.get((cell_x, cell_y))
                (bmin_x, bmin_y, bmax_x, bmax_y) = perVxbounds if perVxbounds else (0, 0, 0, 0)
                if grid[cell_y][cell_x] == 1:
                    temp_x, temp_y = start
                    temp_x *= per_voxel_bounds_scale
                    temp_y *= per_voxel_bounds_scale

                    if ray_intersects_aabb((temp_x, temp_y), direction, (bmin_x, bmin_y), (bmax_x, bmax_y))[0]:
                        hit = True
                        break  # Stop when hitting an 'on' cell
            else:
                if grid[cell_y][cell_x] == 1:
                    hit = True
                    break  # Stop when hitting an 'on' cell
        else:
            out_of_bounds = True
            break  # Out of bounds
       
        # Move to the next grid cell and record the exact intersection point
        intersect_x = 0
        intersect_y = 0
        if tMax_x < tMax_y:
            intersect_x = cell_x + (step_x > 0)  # Exact horizontal edge intersection
            intersect_y = y + (tMax_x * dy)
            cell_x += step_x
            tMax_x += tDelta_x
        else:
            intersect_x = x + (tMax_y * dx)
            intersect_y = cell_y + (step_y > 0)  # Exact vertical edge intersection
            cell_y += step_y
            tMax_y += tDelta_y
       
        # Check bounds if provided
        if bounds:
            min_x, min_y, max_x, max_y = bounds
            if not (min_x <= intersect_x <= max_x and min_y <= intersect_y <= max_y):
                out_of_bounds = True
                break

        steps_taken = steps_taken + 1
        intersected_points.append((intersect_x, intersect_y))
   
    return hit, out_of_bounds, traversed, intersected_points, steps_taken
       
def create_bounding_boxes(buffer, factor=4):
    low_res_shape = (buffer.shape[0] // factor, buffer.shape[1] // factor)
    bounding_boxes = {}
    for y in range(low_res_shape[0]):
        for x in range(low_res_shape[1]):
            min_x = 1e10
            min_y = 1e10
            max_x = -1
            max_y = -1
            for sx in range(factor):
                for sy in range(factor):
                    if buffer[y * factor + sy, x * factor + sx]:
                        min_x = min(min_x, x * factor + sx)
                        min_y = min(min_y, y * factor + sy)
                        max_x = max(max_x, x * factor + sx)
                        max_y = max(max_y, y * factor + sy)
            if min_x <= max_x and min_y <= max_y:
                bounding_boxes[(x, y)] = (min_x, min_y, max_x + 1, max_y + 1)  # Add 1 to max values to make it inclusive

    return bounding_boxes

def display_voxel_buffer(buffer, low_res_buffer, bounding_boxes, factor=4):
    pygame.init()
    global t_screen
    global t_scale
    screen_width, screen_height = 1920, 1080  # Set screen size to a fixed width and height
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('Voxel Buffer Display')
    t_screen = screen

    voxel_width, voxel_height = buffer.shape
    low_res_width, low_res_height = low_res_buffer.shape
    ascale = 2 # Adjust this scale factor to make the voxels bigger
    scale = min(screen_width / voxel_width, screen_height / voxel_height) * ascale
    t_scale = scale
    offset_x, offset_y = 0, 0  # Initial offset
    dragging = False
    last_mouse_pos = None
    mouse_pos = pygame.mouse.get_pos()
    mx = ((mouse_pos[0] - offset_x) / (scale * factor))
    my = ((mouse_pos[1] - offset_y) / (scale * factor))
    running = True
    direction = (1,1)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    dragging = True
                    last_mouse_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    mouse_x, mouse_y = event.pos
                    dx = mouse_x - last_mouse_pos[0]
                    dy = mouse_y - last_mouse_pos[1]
                    offset_x += dx
                    offset_y += dy
                    last_mouse_pos = (mouse_x, mouse_y)

        screen.fill((255, 255, 255))  # Clear screen with white

        # Draw lower resolution buffer
        for y in range(low_res_height):
            for x in range(low_res_width):
                color = (255, 182, 193) if low_res_buffer[y, x] else (173, 216, 230)  # Light pink for filled voxels, light blue for empty voxels
                pos_x = int(np.ceil(offset_x + x * scale * factor))
                pos_y = int(np.ceil(offset_y + y * scale * factor))
                size = int(np.ceil(scale * factor + 1))
                bsize = int(np.ceil(scale * factor + 1))
                pygame.draw.rect(screen, color, (pos_x, pos_y, size, size))

        # Draw higher resolution buffer
        for y in range(voxel_height):
            for x in range(voxel_width):
                low_res_x = x // factor
                low_res_y = y // factor
                if low_res_buffer[low_res_y, low_res_x]:
                    if buffer[y, x]:
                        color = (255, 182, 193)  # Light pink for filled voxels
                        pos_x = int(np.ceil(offset_x + x * scale))
                        pos_y = int(np.ceil(offset_y + y * scale))
                        size = int(np.ceil(scale + 1))
                        pygame.draw.rect(screen, color, (pos_x, pos_y, size, size))
                        pygame.draw.rect(screen, (15, 15, 15), (pos_x, pos_y, size, size), 1)  # Draw light gray border
                    else:
                        color = (255, 255, 153)  # Light yellow for empty voxels
                        pos_x = int(np.ceil(offset_x + x * scale))
                        pos_y = int(np.ceil(offset_y + y * scale))
                        size = int(np.ceil(scale + 1))
                        pygame.draw.rect(screen, color, (pos_x, pos_y, size, size))
                        pygame.draw.rect(screen, (15, 15, 15), (pos_x, pos_y, size, size), 1)

        # Draw lower resolution buffer borders
        for y in range(low_res_height):
            for x in range(low_res_width):
                color = (255, 182, 193) if low_res_buffer[y, x] else (173, 216, 230)  # Light pink for filled voxels, light blue for empty voxels
                pos_x = int(np.ceil(offset_x + x * scale * factor))
                pos_y = int(np.ceil(offset_y + y * scale * factor))
                size = int(np.ceil(scale * factor + 2))
                bsize = int(np.ceil(scale * factor + 2))
                pygame.draw.rect(screen, (15, 15, 15), (pos_x, pos_y, size, size), 2)  # Draw light gray border

        # Draw bounding boxes
        for (x, y), (min_x, min_y, max_x, max_y) in bounding_boxes.items():
            pos_x = int(np.ceil(offset_x + min_x * scale))
            pos_y = int(np.ceil(offset_y + min_y * scale))
            width = int(np.ceil((max_x - min_x) * scale) + 2)
            height = int(np.ceil((max_y - min_y) * scale) + 2)
            pygame.draw.rect(screen, (128, 0, 255), (pos_x, pos_y, width, height), 2)  # Green bounding boxes

        # Draw DDA line
        keys = pygame.key.get_pressed()
        if keys[pygame.K_t]:
            mouse_pos = pygame.mouse.get_pos()
            mx = ((mouse_pos[0] - offset_x) / (scale * factor))
            my = ((mouse_pos[1] - offset_y) / (scale * factor))

        if keys[pygame.K_f]:
            new_mouse_pos = pygame.mouse.get_pos()
            new_mx = ((new_mouse_pos[0] - offset_x) / (scale * factor))
            new_my = ((new_mouse_pos[1] - offset_y) / (scale * factor))
            direction = (new_mx - mx, new_my - my)
            if abs(direction[0]) > 0.01 or abs(direction[1]) > 0.01:
                direction = direction / np.linalg.norm(direction)  # Normalize direction
            else:
                direction = (1, 0)  # Default direction if zero vector

        start = (mx, my)

        # Draw intersect points for low resolution buffer
        # Draw the ray direction line
        end_x = start[0] + direction[0] * 100  # Extend the line to a reasonable length
        end_y = start[1] + direction[1] * 100
        pygame.draw.line(screen, (0, 0, 255),
             (int(np.ceil(offset_x + start[0] * scale * factor)), int(np.ceil(offset_y + start[1] * scale * factor))),
             (int(np.ceil(offset_x + end_x * scale * factor)), int(np.ceil(offset_y + end_y * scale * factor))), 2)

        previous_cell = (-1,-1)
        total_steps = 0

        # Check if start point is out of bounds
        min_x, min_y, max_x, max_y = (0, 0, low_res_width, low_res_height)
        if not (min_x <= start[0] <= max_x and min_y <= start[1] <= max_y):
            # Find intersection with bounds
            hit, intersection_point = ray_intersects_aabb(start, direction, (min_x, min_y), (max_x, max_y))
            if hit:
                start = (intersection_point[0] - 1e-10, intersection_point[1] - 1e-10)

        while(True):
            # Draw DDA line for low resolution buffer
            hit, out_of_bounds, traversed_cells, points, steps_taken = dda_ray_traversal(low_res_buffer, start, direction, per_voxel_bounds=bounding_boxes, per_voxel_bounds_scale=factor)
            total_steps += steps_taken
            # Draw DDA line for high resolution buffer starting from the last point of low resolution DDA or cursor position
            if points:
                start_high_res = (points[-1][0] * factor, points[-1][1] * factor)
            else:
                start_high_res = (start[0] * factor, start[1] * factor)
                # Calculate bounds for high resolution DDA trace
            
            # Draw the intersected points from DDA
            for i, point in enumerate(points):
                point_x, point_y = point
                color = (0, 255, 0) if i == len(points) - 1 else (255, 0, 0)  # Green for the last point, red for others
                radius = 6 if i == len(points) - 1 else 3  # Bigger radius for the last point
                pygame.draw.circle(screen, color,
                (int(np.ceil(offset_x + point_x * scale * factor)), int(np.ceil(offset_y + point_y * scale * factor))),
                radius)  # Draw circles for intersected points

            if hit and not out_of_bounds:
                if traversed_cells:
                    if previous_cell == traversed_cells[-1]:
                        break

                    # Calculate bounds for high resolution DDA trace
                    last_cell_x, last_cell_y = traversed_cells[-1]
                    previous_cell = (last_cell_x, last_cell_y)
                    min_x = last_cell_x * factor
                    min_y = last_cell_y * factor
                    max_x = min_x + factor
                    max_y = min_y + factor
                    bounds = (min_x, min_y, max_x, max_y)
                else:
                    bounds = None
               
                # Draw DDA line for high resolution buffer
                hit, out_of_bounds, traversed_cells_high_res, points_high_res, steps_taken = dda_ray_traversal(buffer, start_high_res, direction, bounds)
                total_steps += steps_taken

                # Draw the intersected points from DDA on high resolution buffer
                for i, point in enumerate(points_high_res):
                    point_x, point_y = point
                    color = (0, 255, 0) if i == len(points_high_res) - 1 else (255, 0, 0)  # Green for the last point, red for others
                    radius = 6 if i == len(points_high_res) - 1 else 3  # Bigger radius for the last point
                    pygame.draw.circle(screen, color,
                    (int(np.ceil(offset_x + point_x * scale)), int(np.ceil(offset_y + point_y * scale))),
                    radius)  # Draw circles for intersected points

                if not hit :
                    if points_high_res:
                        eps = 1e-10
                        start = (points_high_res[-1][0] / factor + direction[0] * eps, points_high_res[-1][1] / factor + direction[1] * eps)
                    else:
                        start = (start_high_res[0] / factor, start_high_res[1] / factor)
                    # Offset start along ray direction back by a small amount
                    continue
                break                
            else:
                break
        
        # Render total steps taken as text on the screen
        font = pygame.font.SysFont(None, 36)
        text_surface = font.render(f"Total steps taken: {total_steps}", True, (0, 0, 0))
        screen.blit(text_surface, (10, 10))
        text_surface = font.render("Press 't' to set the start point", True, (0, 0, 0))
        screen.blit(text_surface, (10, 46))
        text_surface = font.render("Press 'f' to set the end point", True, (0, 0, 0))
        screen.blit(text_surface, (10, 76))
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    shape = (256, 256)
    chunk_size = 8
    voxel_buffer = create_voxel_buffer(shape)
    low_res_buffer = create_lower_resolution_buffer(voxel_buffer, chunk_size)
    bounding_boxes = create_bounding_boxes(voxel_buffer, factor=chunk_size)  # Create bounding boxes for the high resolution buffer
    display_voxel_buffer(voxel_buffer, low_res_buffer, bounding_boxes, chunk_size)