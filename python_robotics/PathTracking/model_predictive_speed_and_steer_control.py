"""
Path tracking simulation with iterative linear model predictive control for speed and steer control

Tutorial by Atsushi Sakai (@Atsushi_twi)

"""

import matplotlib.pyplot as plt
import time
import cvxpy
import math
import numpy as np
import pathlib
# sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from utils.angle import angle_mod

from PathPlanning.CubicSpline import cubic_spline_planner

# State and control dimensions
_s_dim = 4 # x = x, y, v, yaw
_u_dim = 2 # u = acc, steer
# Horizon Length
N = 5

# MPC Parameters
R = np.diag([0.01, 0.01])         # input cost matrix
Rd = np.diag([0.01, 1.0])         # input difference cost matrix
Q = np.diag([1.0, 1.0, 0.5, 0.5]) # state cost matrix
Qf = Q                            # state final matrix
GOAL_DIST = 1.5                   # goal distance
STOP_SPEED = 0.5 / 3.6            # stop speed
MAX_TIME = 500.0                  # Max simulation time

# iterative parameter
MAX_ITER = 3                      # Max iteration
DU_TH = 0.1                       # iteration finish param (measures change in control signal between iterations)

TARGET_SPEED = 10.0 / 3.6         # target speed [m/s]
N_IND_SEARCH = 10                 # Search index number

DT = 0.2                          # time tick [s]

# Vehicle Parameters
LENGTH = 4.5
WIDTH = 2.0
BACKTOWHEEL = 1.0
WHEEL_LEN = 0.3
WHEEL_WIDTH = 0.2
TREAD = 0.7
WB = 2.5

MAX_STEER = np.deg2rad(45.0)      # maximum steering angle [rad]
MAX_STEER_DOT = np.deg2rad(30.0)  # maximum steering rate [rad/s]
MAX_SPEED = 55.0 / 3.6            # maximum speed [m/s]
MIN_SPEED = -20 / 3.6             # minimum speed [m/s]
MAX_ACCEL = 1.0                   # maximum acceleration [m/s^2]

show_animation = True


class State:
    """
    Vehicle state class
    """
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.preDelta = None

def pi_2_pi(angle):
    return angle_mod(angle)

def get_linear_model_matrix(v, phi, delta):

    A = np.array([[1.0,              0.0,   DT*math.cos(phi), -DT*v*math.sin(phi)],
                  [0.0,              1.0,   DT*math.sin(phi),  DT*v*math.cos(phi)],
                  [0.0,              0.0,                1.0,                 0.0],
                  [0.0,              0.0, DT*math.tan(delta),                 1.0]])
    
    B = np.array([[0.0,                          0.0],
                  [0.0,                          0.0],
                  [ DT,                          0.0],
                  [0.0, DT*v/(WB*math.cos(delta)**2)]])
    
    C = np.array([ DT * v * math.sin(phi) * phi,
                  -DT * v * math.cos(phi) * phi,
                  -DT * v * delta / (WB * math.cos(delta) ** 2)])
    
    return A, B, C

def plot_car(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):
    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, -WIDTH / 2, -WIDTH / 2, WIDTH /2]])
    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])
    
    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(fl_wheel)

    Rot1 = np.array([[   math.cos(yaw),   math.sin(yaw)],
                     [  -math.sin(yaw),   math.cos(yaw)]])
    
    Rot2 = np.array([[ math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])
    
    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
            np.array(fl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
            np.array(rr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
            np.array(rl_wheel[1, :]).flatten(), truckcolor)
    

def update_state(state, a, delta):
    # input check
    if delta >= MAX_STEER:
        delta = MAX_STEER
    elif delta <= -MAX_STEER:
        delta = -MAX_STEER

    state.x = state.x + state.v * math.cos(state.yaw) * DT
    state.y = state.y + state.v * math.sin(state.yaw) * DT
    state.yaw = state.yaw + state.v / WB * math.tan(delta) * DT
    state.v = state.v + a * DT

    if state.v > MAX_SPEED:
        state.v = MAX_SPEED
    elif state.v < MIN_SPEED:
        state.v = MIN_SPEED

    return state

def get_nparray_from_matrix(x):
    return np.array(x).flatten()

def calc_nearest_index(state, cx, cy, cyaw, pind):

    dx = [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
    dy = [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    min_d = min(d)

    ind = d.index(min_d) + pind

    min_d = math.sqrt(min_d)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        min_d *= -1

    return ind, min_d

def predict_motion(x0, oa, od, xref):
    xbar = xref * 0.0
    for i, _ in enumerate(x0):
        xbar[i, 0] = x0[i]

    state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
    for (ai, di, i) in zip(oa, od, range (1, N + 1)):
        state = update_state(state, ai, di)
        xbar[0, i] = state.x
        xbar[1, i] = state.y
        xbar[2, i] = state.v
        xbar[3, i] = state.yaw

    return xbar

def iterative_linear_mpc_control(xref, x0, dref, oa, od):
    """
    MPC contorl with updatind operational point iteratively
    """
    ox, oy, oyaw, ov = None, None, None, None

    if oa is None or od is None:
        oa = [0.0] * N
        od = [0.0] * N

    for i in range(MAX_ITER):
        xbar = predict_motion(x0, oa, od, xref)
        poa, pod = oa[:], od[:]
        oa, od, ox, oy, oyaw, ov = linear_mpc_control(xref, xbar, x0, dref)
        du = sum(abs(oa - poa)) + sum(abs(od - pod)) # calulate u change value
        if du <= DU_TH: break
    
    else: 
        print("Iterative is max iter")

    return oa, od, ox, oy, oyaw, ov

def linear_mpc_control(xref, xbar, x0, dref):
    """
    linear mpc control
    --------------------
    xref: reference point
    xbar: operational point
    x0: initial state
    dref: reference steering angle
    """

    x = cvxpy.Variable((_s_dim, N + 1))
    u = cvxpy.Variable((_u_dim, N))

    cost = 0.0

    constraints = []

    for k in range(N):
        cost += cvxpy.quad_form(u[:, k], R)

        if k != 0:
            cost += cvxpy.quad_form(xref[:, k] - x[:, k], Q)

        A, B, C = get_linear_model_matrix(
            xbar[2, k], xbar[3, k], dref[0, k]
        )

        constriants += [x[:, k+1] == A @ x[:, k] + B @ u[:, k] + C] # x_k+1 = Ax_k + Bu_k + C

        if k < (N-1):
            cost += cvxpy.quad_form(u[:, k+1] - u[:, k], Rd)
            constraints += [cvxpy.abs(u[1, k+1] - u[1, k]) <= MAX_STEER_DOT * DT]

    cost += cvxpy.quad_form(xref[:, N] - x[:, N], Qf)

    constraints += [x[:, 0] == x0] # initial condition constriant
    constraints += [x[2, :] <= MAX_SPEED]
    constraints += [x[2, :] >= MIN_SPEED]
    constraints += [cvxpy.abs(u[0, :]) <= MAX_ACCEL]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.CLARABEL, verbose=False)


