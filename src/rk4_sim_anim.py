
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# Read CSV file
# ============================================================

csv_file = "/Users/ankitpanda/Downloads/solar_system_data2_80yr.csv"

df = pd.read_csv(csv_file)

print("CSV file loaded successfully!")
print("Number of data points =", len(df))
print("Number of columns =", len(df.columns))


# ============================================================
# Planet names
# ============================================================

names = [
    "Sun",
    "Mercury",
    "Venus",
    "Earth",
    "Mars",
    "Jupiter",
    "Saturn",
    "Uranus",
    "Neptune",
    "Pluto"
]

n = len(names)

AU = 1.495978707e11


# ============================================================
# Extract time
# ============================================================

time_years = df["Time_years"].values


# ============================================================
# Extract positions
# Convert metres -> AU
# ============================================================

positions = np.zeros((len(df), n, 3))

for i in range(n):

    positions[:, i, 0] = df[
        f"{names[i]}_x_m"
    ].values / AU

    positions[:, i, 1] = df[
        f"{names[i]}_y_m"
    ].values / AU

    positions[:, i, 2] = df[
        f"{names[i]}_z_m"
    ].values / AU


# ============================================================
# Extract velocities
# ============================================================

velocities = np.zeros((len(df), n, 3))

for i in range(n):

    velocities[:, i, 0] = df[
        f"{names[i]}_vx_m_s"
    ].values

    velocities[:, i, 1] = df[
        f"{names[i]}_vy_m_s"
    ].values

    velocities[:, i, 2] = df[
        f"{names[i]}_vz_m_s"
    ].values


# ============================================================
# Reduce number of animation frames
# ============================================================

# Change this value depending on animation size
frame_step = 500
#frame_step = 20

frame_indices = np.arange(
    0,
    len(df),
    frame_step
)

print("Number of animation frames =", len(frame_indices))


# ============================================================
# Create 3D figure
# ============================================================

fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(
    111,
    projection="3d"
)


# ============================================================
# Determine plot limits
# ============================================================

max_distance = np.max(
    np.abs(positions)
)

plot_limit = max_distance * 1.05

ax.set_xlim(
    -plot_limit,
    plot_limit
)

ax.set_ylim(
    -plot_limit,
    plot_limit
)

ax.set_zlim(
    -plot_limit,
    plot_limit
)


# ============================================================
# Labels
# ============================================================

ax.set_xlabel("X (AU)")
ax.set_ylabel("Y (AU)")
ax.set_zlabel("Z (AU)")

ax.set_title(
    "Solar System N-Body Simulation\n"
    "Centre of Mass Frame"
)


# ============================================================
# Create orbit lines
# ============================================================

orbit_lines = []

for i in range(n):

    line, = ax.plot(
        [],
        [],
        [],
        label=names[i]
    )

    orbit_lines.append(line)


# ============================================================
# Create planet points
# ============================================================

planet_points = []

for i in range(n):

    point, = ax.plot(
        [],
        [],
        [],
        marker="o",
        linestyle="",
        markersize=5
    )

    planet_points.append(point)


# ============================================================
# Time text
# ============================================================

time_text = ax.text2D(
    0.05,
    0.95,
    "",
    transform=ax.transAxes
)


# ============================================================
# Velocity text
# ============================================================

velocity_text = ax.text2D(
    0.05,
    0.88,
    "",
    transform=ax.transAxes,
    fontsize=9
)


# ============================================================
# Animation initialization
# ============================================================

def init():

    for i in range(n):

        orbit_lines[i].set_data(
            [],
            []
        )

        orbit_lines[i].set_3d_properties(
            []
        )

        planet_points[i].set_data(
            [],
            []
        )

        planet_points[i].set_3d_properties(
            []
        )

    time_text.set_text("")

    velocity_text.set_text("")

    return (
        orbit_lines
        + planet_points
        + [time_text, velocity_text]
    )


# ============================================================
# Animation update
# ============================================================

def update(frame_number):

    index = frame_indices[frame_number]

    # --------------------------------------------------------
    # Draw orbit up to current position
    # --------------------------------------------------------

    for i in range(n):

        orbit_lines[i].set_data(
            positions[
                :index + 1,
                i,
                0
            ],
            positions[
                :index + 1,
                i,
                1
            ]
        )

        orbit_lines[i].set_3d_properties(
            positions[
                :index + 1,
                i,
                2
            ]
        )

        # ----------------------------------------------------
        # Current planet position
        # ----------------------------------------------------

        planet_points[i].set_data(
            [positions[index, i, 0]],
            [positions[index, i, 1]]
        )

        planet_points[i].set_3d_properties(
            [positions[index, i, 2]]
        )

    # --------------------------------------------------------
    # Display simulation time
    # --------------------------------------------------------

    time_text.set_text(
        f"Simulation time = "
        f"{time_years[index]:.2f} years"
    )

    # --------------------------------------------------------
    # Display velocities
    # --------------------------------------------------------

    velocity_string = ""

    for i in range(1, n):

        velocity_magnitude = np.linalg.norm(
            velocities[index, i]
        )

        velocity_string += (
            f"{names[i]}: "
            f"{velocity_magnitude:.2f} m/s\n"
        )

    velocity_text.set_text(
        velocity_string
    )

    return (
        orbit_lines
        + planet_points
        + [time_text, velocity_text]
    )


# ============================================================
# Create animation
# ============================================================

animation = FuncAnimation(
    fig,
    update,
    frames=len(frame_indices),
    init_func=init,
    interval=30,
    blit=False
)


# ============================================================
# Legend
# ============================================================

ax.legend(
    loc="upper right"
)

animation.save(
    "/Users/ankitpanda/Downloads/solar_system_animation.mp4",
    writer="ffmpeg",
    fps=30,
    dpi=150
)
# ============================================================
# Show animation
# ============================================================

plt.show()