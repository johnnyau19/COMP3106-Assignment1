# Name this file to assignment1.py when you submit
import csv
import os
from queue import PriorityQueue
from typing import List, Tuple, Set, Dict


# Manhattan distance is used by heuristic function
def manhattan(start_coord : Tuple[int, int], end_coord : Tuple[int, int]) -> int:
    mahatten_dist = abs(start_coord[0] - end_coord[0]) + abs(start_coord[1] - end_coord[1])
    return mahatten_dist

# Heuristic function approximates the distance
def heuristic(state : Tuple[int, int, int, Tuple[int,...]], treasure_coords : List[Tuple[int,int]], goal_coords : Tuple[int, int]) -> int:
    curr_coord = state[:2]
    treasure_value = state[2]

    # We change the heuristic method once we have achieved >= 5 as a treasure value
    if treasure_value >= 5:                           
        dist_goal = []
        for goal in goal_coords:
            dist_goal.append(manhattan(curr_coord, goal))
        return min(dist_goal)
    
    dist = []
    for treasure in treasure_coords:
        for goal in goal_coords:
            dist.append(manhattan(curr_coord, treasure) + manhattan(treasure, goal))
    return min(dist)


# Find the adjacent squares that we can move to
def successors(state : Tuple[int, int, int, Tuple[int,...]], walls : Set[Tuple[int,int]], grid_shape : Tuple[int,int]) -> List[Tuple[int, int]]:
    curr_coord = state[:2]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    children = []
    for direction in directions:
        new_coord = (curr_coord[0] + direction[0], curr_coord[1] + direction[1])
        if new_coord not in walls and 0 <= new_coord[0] < grid_shape[0] and 0 <= new_coord[1] < grid_shape[1]:
            children.append(new_coord)
    return children

# Preprocess the file, extract information from the grid such as positions of treasures, walls, and size.
def file_preprocess(filepath : str) -> Tuple[Tuple[int,int], Set[Tuple[int,int]], Set[Tuple[int,int]], Dict[Tuple[int,int], int], Tuple[int,int]]: 
  start = None
  goals = set()
  walls = set()
  treasures = {}
  total_treasure = 0
  grid_shape = None
  
  with open(os.path.abspath(filepath), newline='') as f:
      reader = csv.reader(f)
      rows, cols = 0, 0
      for row_idx, row in enumerate(reader):
          rows = row_idx+1
          for col_idx, cell in enumerate(row):
              cols = col_idx+1
              if cell == "S":
                  start = (row_idx, col_idx)
              elif cell == "G":
                  goals.add((row_idx, col_idx))
              elif cell == "X":
                  walls.add((row_idx, col_idx))
              elif cell.isdigit() and 1 <= int(cell) <= 5:
                  treasures[(row_idx, col_idx)] = int(cell)
                  total_treasure += int(cell)
      grid_shape = (rows,cols)
#   print("Start:", start)
#   print("Goals:", goals)
#   print("Walls:", walls)
#   print("Treasures:", treasures)
#   print("Total treasure:", total_treasure)
#   print("Size: ", grid_shape)
  return start, goals,walls, treasures, grid_shape



# The pathfinding function must implement A* search to find the goal state
def pathfinding(filepath="Examples/Examples/Example2/grid.txt") -> Tuple[List[Tuple[int,int]], int, int]:
  # Boolean value to inform if the optimal path is found
  path_is_found=False

  # Preprocess the grid
  start, goal_coords, walls, treasures_coords, grid_shape = file_preprocess(filepath)
  
  # Prepare frontier and explored set 
  explored = {}
  frontier = PriorityQueue()

  # Move the start state to the frontier first
  state = (start[0], start[1], 0, ())    # we also consider the treasure_value as part of the state -> (row, column, treasure_value, list of visited treasures' position)

  g_val = 0 
  h_val = heuristic(state, treasures_coords, goal_coords)
  f_val = h_val + g_val

  # Set a counter to prevent priority queue from comparing elements by using start_state, since it compares from left to right
  counter = 0 
  frontier.put((f_val, h_val, g_val, counter, state, None))



  while frontier.qsize() > 0: 
      leaf = frontier.get()    # pop out the smallest from the queue
      state = leaf[4]

      curr_coord, treasure_value = state[:2], state[2]

      # If the current state is a goal state (r_goal, c_goal, treasure_value >= 5), then we finish exploring 
      if curr_coord in goal_coords and treasure_value >=5:
          explored[state] = leaf[:3] + (leaf[5],) 
          path_is_found = True
          break

      # We cannot revisit a state that was visited since f(n) always increase along the exploration
      if state in explored:
          continue 
      
      # Update the treasure_value once we are exploring the state/coordination which has a treasure
      if curr_coord in treasures_coords and curr_coord not in state[3]:
          state = (curr_coord[0], curr_coord[1], treasure_value + treasures_coords.get(curr_coord), state[3] + (curr_coord,))     # update the visited treasures' position list, since we are standing at where treasure is


      explored[state] = leaf[:3] + (leaf[5],)     #  { (r,c, treasure_value, list of visited treasures' position) : (f_val, h_val, g_val, parent_state) }


      # Relaxation
      for adjacent_square_coord in successors(state, walls, grid_shape):
          # Since we haven't officially explore move to / visit this state, we keep the same treasure_value as current
          next_possible_state = (adjacent_square_coord[0], adjacent_square_coord[1], state[2], state[3])            
          
          # If this next state is not explored yet, then we continue the calculation, otherwise we skip it
          if next_possible_state in explored:
              continue
          
          # If the next state hasn't been explored
          g_val = explored.get(state)[2] + 1                                 # Move from current state to the next state with the cost = 1
          h_val = heuristic(next_possible_state, treasures_coords, goal_coords)      # Calculate f, h values
          f_val = h_val + g_val
          counter += 1    
          frontier.put((f_val, h_val, g_val, counter, next_possible_state, state))



  # optimal_path_cost is the cost of the optimal path
  optimal_path_cost = explored.get(state)[2] if path_is_found else None

  # optimal_path is a list of coordinate of squares visited (in order)
  optimal_path = []
  if path_is_found:
    while state is not None:
        coord = state[:2]
        optimal_path.append(coord)
        state = explored.get(state)[3]    # Get the parent state
    
    optimal_path.reverse()

  # num_states_explored is the number of states explored during A* search
  num_states_explored = len(explored) 

  return optimal_path, optimal_path_cost, num_states_explored

print(pathfinding())
