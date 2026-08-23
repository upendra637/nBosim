import numpy as np

import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from model import Nbody


# initial conditions for the N-body simulation

# masses in solar masses
masses = np.array([
    1.0,
    1.660e-7,
    2.448e-6,
    3.003e-6,
    3.227e-7,
    9.545e-4,
    2.857e-4,
    4.366e-5,
    5.151e-5
])

# average orbital distances from Sun in AU
X = np.array([
    0.0,
    0.387,
    0.723,
    1.000,
    1.524,
    5.203,
    9.537,
    19.191,
    30.07
])

# y positions (initially zero)
y = np.zeros(len(masses))

# x velocities (initially zero)
ux = np.zeros(len(masses))

# y velocities
uy = np.array([
    0.0,
    10.07,
    7.39,
    6.28,
    5.09,
    2.62,
    2.12,
    1.48,
    1.14
])

dt = 0.01

nbody = Nbody(
    masses,
    X,
    y,
    ux,
    uy,
    dt
)


N = 10000

x_pos = np.zeros((len(masses), N))
y_pos = np.zeros((len(masses), N))


print("Running simulation...")

for i in range(N):

    X, Y = nbody.position()

    x_pos[:, i] = X
    y_pos[:, i] = Y


print("Simulation finished")


# Use every 20th simulation step

step = 20

frames = range(0, N, step)

print("Number of animation frames:", len(frames))



fig, ax = plt.subplots(figsize=(16, 8))

ax.set_xlim(-35, 35)
ax.set_ylim(-35, 35)

ax.set_aspect("equal")

ax.set_xlabel("X Position (AU)")
ax.set_ylabel("Y Position (AU)")

ax.set_title("N-Body Simulation")

ax.grid()

bodies, = ax.plot(
    [],
    [],
    "o",
    markersize=5
)

trails = []

for i in range(len(masses)):

    trail, = ax.plot(
        [],
        [],
        linewidth=1
    )

    trails.append(trail)

def update(frame):

    # Current positions
    bodies.set_data(
        x_pos[:, frame],
        y_pos[:, frame]
    )

    # Trails
    for i in range(len(masses)):

        trails[i].set_data(
            x_pos[i, :frame + 1],
            y_pos[i, :frame + 1]
        )

    return [bodies] + trails


print("Creating animation...")

animation = FuncAnimation(
    fig,
    update,
    frames=frames,
    interval=20,
    blit=True
)

print("Animation object created")


print("Saving MP4...")

writer = FFMpegWriter(
    fps=30,
    bitrate=1200
)

animation.save(
    "nbody_animation.mp4",
    writer=writer
)

print("Animation saved successfully!")