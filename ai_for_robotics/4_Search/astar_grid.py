""" 
This collection of files is based on daniel-s-ingram's visualizations 
of the course "Artificial Intelligence for Robotics" by Sebastian Thrun

This particular file focuses on discret search, using A* with a distance heuristic ic through a 2D environment.
"""

# Import
from __future__ import print_function, division
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#
GRID_LEN_Y = 100            # Rows
GRID_LEN_X = 100            # Columns
OBSTACLE_PROBABILITY = 0.25 # % of grid space as obstacles

# Initialize grid
GRID = np.int8(np.random((GRID_LEN_Y, GRID_LEN_X) > (1 - OBSTACLE_PROBABILITY)))
START_NODE = (5, 5)
GOAL_NODE = (GRID_LEN_Y - 5, GRID_LEN_X - 5)
X, Y = np.meshgrid([i for i in range(GRID_LEN_X)], [i for i in range(GRID_LEN_Y)])
HEURISTIC = np.abs(X - GOAL_NODE[1]) + np.abs(Y - GOAL_NODE[0])

visited = np.zeros(GRID_LEN_Y, GRID_LEN_X)
visited[START_NODE] = 1
expand = -np.ones((GRID_LEN_Y, GRID_LEN_X))
action = -np.ones((GRID_LEN_Y, GRID_LEN_X), dtype=np.int8)
x = START_NODE[0]
y = START_NODE[1]
g = 0                      # Cost from start
h = HEURISTIC[x, y]        # Expected cost to goal
f = g + h                  # Predicted cost of path through a cell

DELTAS = [[1, 0],          # Actions
          [0, 1],
          [-1, 0],
          [0, -1]]

open = [[f, g, x, y]]      # Queue of next cells to visit

found = False              # Flag for if a path has been found
resign = False             # Flag for if no path exists
count = 0                  # Order of visted nodes
cost = 1                   # Minimum cost to goal

#
fig, ax = plt.subplots()
route, = plt.plot([], [], 'b-')

def init():
    ax.imshow(np.logical_not(GRID), cmap="gray")
    ax.plot(START_NODE[1], START_NODE[0], 'ro')
    ax.plot(GOAL_NODE[1], GOAL_NODE[0], 'g*')
    return route,

def get_current_route(current_node):
    route = [current_node]
    while route[0] != START_NODE:
        node = route[0]
        delta = DELTAS[action[node]]
        previous_node = (node[0] - delta[0], node[1] - delta[1])
        route.insert(0, previous_node)
    return np.array(route)

def animate(_):
    global count, open
    if len(open) == 0:
        resign = True
        return route,
    else:
        open.sort()
        open.reverse()
        next = open.pop()
        x = next[2]
        y = next[3]
        g = next[1]
        expand[x, y] = count
        count += 1

        if (x, y) == GOAL_NODE:
            found = True
            open = []
        else:
            for i in range(len(DELTAS)):
                x1 = x + DELTAS[i][0]
                y1 = y + DELTAS[i][1]
                if x1 >= 0 and x1 < GRID_LEN_Y and y1 >= 0 and y1 < GRID_LEN_X:
                    if visited[x1, y1] == 0 and GRID[x1, y1] == 0:
                        g1 = g + cost
                        f1 = HEURISTIC[x1, y1]
                        open.append([f1, g1, x1, y1])
                        visited[x1, y1] = 1
                        action[x1, y1] = i
            current_route = get_current_route((x, y))
            x = current_route[:, 1]
            y = current_route[:, 0]
            route.set_data(x, y)
        return route,

anim = animation.FuncAnimation(fig, animate, 350, interval=50, init_func=init)
plt.show()
#anim.save("a_star.gif", writer="imagemagick")