# Name this file to assignment1.py when you submit

# The pathfinding function must implement A* search to find the goal state
def pathfinding(filepath):
  # filepath is the path to a CSV file containing a grid 

  # optimal_path is a list of coordinate of squares visited (in order)
  # optimal_path_cost is the cost of the optimal path
  # num_states_explored is the number of states explored during A* search
  return optimal_path, optimal_path_cost, num_states_explored


def manhatten(start_coord, end_coord):
    mahatten_dist = [start_coord[0] - end_coord[0], start_coord[1] - end_coord[1]]
    return mahatten_dist


def heuristic(curr_coord, treasure_total, treasure_coords, goal_coords):
    if treasure_total >= 5:
        dist_goal = []
        for goal in goal_coords:
            dist_goal.append(manhatten(curr_coord, goal))
        return min(dist_goal)
    
    dist = []
    for treasure in treasure_coords:
        for goal in goal_coords:
            dist.append(manhatten(curr_coord, treasure) + manhatten(treasure, goal))

    return min(dist)


def successors(curr_coord):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    children = []
    for direction in directions:
        new_coord = (curr_coord[0] + direction[0], curr_coord[1] + direction[1])
        if new_coord not in walls and 0 <= new_coord[0] < grid_size[0] and 0 <= new_coord[1] < grid_size[1]:
            children.append(new_coord)
    return children
