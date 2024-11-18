""" 
This collection of files is based on daniel-s-ingram's visualizations 
of the course "Artificial Intelligence for Robotics" by Sebastian Thrun

This particular file focuses on probablistic 1D localization.
"""
# Imports
from __future__ import print_function, division
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import from_levels_and_colors

# Number of grid cells
n_cells = 100

# Sensing distribution
p_hit = 0.6               # Sensor reading is correct
p_miss = 0.2              # Sensor reading is incorrect
# Movement distribution
p_exact = 0.8             # Robot's movement is exactly the desired
p_undershoot = 0.1        # Robot's movement is less than the desired
p_overshoot = 0.1         # Robot's movement is greater than the desired

# Start with uniform probability distribution
p = [1/n_cells for _ in range(n_cells)]

### If Starting position is known ###
# start = 50
# p = [0 for _ in range(n_cells)]
# p[start] = 1

landmarks = [1 if random.random() > 0.8 else 0 for _ in range(n_cells)] # ~ 1/5 landmarks are 1, others are zero
colors = ["white", "black"]
levels = [0, 1, 2]
cmap, norm = from_levels_and_colors(levels, colors)

def sense(p, Z):
    q = []
    for i in range(n_cells):
        hit = (landmarks[i] == Z)
        if hit:
            q.append(p[i] * p_hit)
        else:
            q.append(p[i] * p_miss)

    # Normalize new distribution
    norm = sum(q)
    q = [q[i]/norm for i in range(n_cells)]
    return q

def move(p, U):
    q = []
    for i in range(n_cells):
        s = p_exact * p[(i-U)%len(p)]
        s += p_undershoot * p[(i-(U-1))%len(p)]
        s += p_overshoot * p[(i-(U+1))%len(p)]
        q.append(s)
    return q

fig, (prob_ax, world_ax) = plt.subplots(
    nrows=2,
    ncols=1,
    gridspec_kw={'height_ratios' : [10, 1]},
    sharex=True)
world_ax.imshow([landmarks, landmarks], cmap=cmap, norm=norm, interpolation=None)
world_ax.get_yaxis().set_ticks([])
x = [i for i in range(n_cells)]
rects = prob_ax.bar(x, p)
line, = world_ax.plot(0, 1, 'r.')

def update(i):
    global p
    p = sense(p, landmarks[i%n_cells])
    print("Sensed P: \n", p)
    p = move(p, 1)
    print("Moved P: \n", p)

    # Update the histogram
    for rect, h in zip(rects, p):
        rect.set_height(h)

    # Update the position graph
    line.set_xdata([(i+1)%n_cells])
    line.set_ydata([0.5])

    # Pause between animation frames
    # input()

    return rects, line,

def init():
    prob_ax.set_xlim(0, n_cells-1)
    prob_ax.set_ylim(0, 1)
    world_ax.set_xlabel("Position")
    prob_ax.set_ylabel("Probability")
    prob_ax.set_title("Histogram Localization with Measurements")
    return rects, line,

anim = animation.FuncAnimation(fig, update, n_cells, interval=50, init_func=init)
plt.show()
