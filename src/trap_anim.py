
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path
import re

# ============================================================
# 1. READ CSV FILE
# ============================================================

file_path = (
    "/Users/ankitpanda/Desktop/"
    "trappist_model_data/TRAPPIST1_simulation_new.csv"
)

df = pd.read_csv(file_path)

print("Number of data rows:", len(df))
print("CSV columns:")
print(df.columns.tolist())

if len(df) < 2:
    raise ValueError("The CSV must contain at least two time steps.")

# ============================================================
# 2. FIND POSITION COLUMNS
# ============================================================

def find_position_columns(columns):

    found = {}

    for col in columns:

        match = re.match(
            r"^(.+)_([xyz])(?:_(m|AU))?$",
            col.strip(),
            flags=re.IGNORECASE
        )

        if match:

            body = match.group(1).strip()
            axis = match.group(2).lower()
            unit = (match.group(3) or "").upper()

            if body not in found:
                found[body] = {}

            # Prefer AU columns over metre columns.
            if axis not in found[body]:
                found[body][axis] = (col, unit)

            elif unit == "AU":
                found[body][axis] = (col, unit)

    return {
        body: data
        for body, data in found.items()
        if all(axis in data for axis in ["x", "y", "z"])
    }


position_columns = find_position_columns(df.columns)

# Remove the COM from the list of physical bodies.
# COM is not a planet.
position_columns = {
    body: cols
    for body, cols in position_columns.items()
    if "com" not in body.lower()
}

if not position_columns:
    raise ValueError("No complete X, Y, Z position columns were found.")

# ============================================================
# 3. EXTRACT POSITIONS IN AU
# ============================================================

AU = 1.495978707e11
SECONDS_PER_YEAR = 365.25 * 24 * 3600

positions = {}

for body, cols in position_columns.items():

    positions[body] = {}

    for axis in ["x", "y", "z"]:

        col, unit = cols[axis]
        values = df[col].to_numpy(dtype=float)

        if unit != "AU":
            values = values / AU

        positions[body][axis] = values

# Identify the central star.
star_candidates = [
    body for body in positions
    if "trappist" in body.lower() or "star" in body.lower()
]

star_name = star_candidates[0] if star_candidates else None

if star_name is None:
    raise ValueError(
        "Could not identify the central star. "
        "Check the position column names."
    )

planet_names = [
    body for body in positions
    if body != star_name
]

print("\nCentral star:", star_name)
print("Planets:", planet_names)

# ============================================================
# 4. FIND VELOCITY COLUMNS
# ============================================================

def find_velocity_columns(columns):

    found = {}

    for col in columns:

        # Examples:
        # TRAPPIST-1 b_vx_m_s
        # TRAPPIST-1 b_vy_m_s
        # TRAPPIST-1 b_vz_m_s
        # TRAPPIST-1 b_vx_AU_yr

        match = re.match(
            r"^(.+)_v([xyz])(?:_(.*))?$",
            col.strip(),
            flags=re.IGNORECASE
        )

        if match:

            body = match.group(1).strip()
            axis = match.group(2).lower()
            unit = (match.group(3) or "").lower()

            if body not in found:
                found[body] = {}

            found[body][axis] = (col, unit)

    return {
        body: data
        for body, data in found.items()
        if all(axis in data for axis in ["x", "y", "z"])
    }


velocity_columns = find_velocity_columns(df.columns)

# ============================================================
# 5. CALCULATE INSTANTANEOUS SPEEDS
# ============================================================

speeds = {}

for body in planet_names:

    if body not in velocity_columns:

        print(
            f"WARNING: Velocity columns not found for {body}. "
            "Speed will be shown as unavailable."
        )

        speeds[body] = None
        continue

    velocity_components = []

    for axis in ["x", "y", "z"]:

        col, unit = velocity_columns[body][axis]

        values = df[col].to_numpy(dtype=float)

        # Convert AU/year to m/s if the column says AU/year.
        if "au" in unit and ("yr" in unit or "year" in unit):
            values = values * AU / SECONDS_PER_YEAR

        # Otherwise, this code assumes velocity is already in m/s.
        # Check your column suffix if your data uses other units.
        velocity_components.append(values)

    vx, vy, vz = velocity_components

    speeds[body] = np.sqrt(vx**2 + vy**2 + vz**2)

# ============================================================
# 6. TIME INFORMATION
# ============================================================

time_column = next(
    (
        col for col in [
            "Time_years",
            "Time_yr",
            "Time_days",
            "Time_seconds",
            "Time_s",
            "Time"
        ]
        if col in df.columns
    ),
    None
)

if time_column is None:

    time_values = np.arange(len(df))
    time_unit = "steps"

else:

    time_values = df[time_column].to_numpy(dtype=float)

    if "day" in time_column.lower():

        time_values = time_values / 365.25
        time_unit = "years"

    elif "second" in time_column.lower() or time_column.lower() == "time_s":

        time_values = time_values / SECONDS_PER_YEAR
        time_unit = "years"

    elif "year" in time_column.lower() or "_yr" in time_column.lower():

        time_unit = "years"

    else:

        time_unit = "simulation time"

# ============================================================
# 7. CREATE THE 3D FIGURE
# ============================================================

fig = plt.figure(figsize=(14, 9), facecolor="black")

ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("black")

ax.set_xlabel("X position (AU)", color="white")
ax.set_ylabel("Y position (AU)", color="white")
ax.set_zlabel("Z position (AU)", color="white")

ax.tick_params(colors="white")

ax.set_title(
    "TRAPPIST-1: 3D Planetary Motion and Velocity",
    color="white",
    fontsize=14
)

ax.xaxis.set_pane_color((0, 0, 0, 1))
ax.yaxis.set_pane_color((0, 0, 0, 1))
ax.zaxis.set_pane_color((0, 0, 0, 1))

# Axis limits use the complete data range, but no orbit is
# drawn beforehand.
all_x = np.concatenate([positions[b]["x"] for b in positions])
all_y = np.concatenate([positions[b]["y"] for b in positions])
all_z = np.concatenate([positions[b]["z"] for b in positions])

limit = 1.1 * max(
    np.max(np.abs(all_x)),
    np.max(np.abs(all_y)),
    np.max(np.abs(all_z)),
    1e-6
)

ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
ax.set_zlim(-limit, limit)
ax.set_box_aspect((1, 1, 1))

# ============================================================
# 8. CREATE MOVING STAR, PLANETS, AND EMPTY TRAILS
# ============================================================

star_marker, = ax.plot(
    [], [], [],
    marker="*",
    color="gold",
    markersize=13,
    linestyle="None",
    label=star_name
)

planet_markers = {}
trail_lines = {}

for body in planet_names:

    # Moving planet marker.
    marker, = ax.plot(
        [], [], [],
        marker="o",
        markersize=5,
        linestyle="None",
        label=body
    )

    planet_markers[body] = marker

    # Empty trail at the start.
    # The trail will grow as the animation progresses.
    trail, = ax.plot(
        [], [], [],
        linewidth=1.0,
        alpha=0.9
    )

    trail_lines[body] = trail

ax.legend(
    loc="upper left",
    bbox_to_anchor=(0.02,0.75),
    facecolor="black",
    labelcolor="white",
    fontsize=8
)

time_text = ax.text2D(
    0.02, 0.93, "",
    transform=ax.transAxes,
    color="white",
    fontsize=11
)

# Velocity display in the right-hand panel.
velocity_text = fig.text(
    0.73, 0.80, "",
    color="white",
    fontsize=10,
    verticalalignment="top",
    family="monospace"
)

fig.subplots_adjust(right=0.72)

# ============================================================
# 9. UPDATE POSITIONS, TRAILS, AND VELOCITIES
# ============================================================

def update(frame):

    # Update the central star's current position.
    star_marker.set_data_3d(
        [positions[star_name]["x"][frame]],
        [positions[star_name]["y"][frame]],
        [positions[star_name]["z"][frame]]
    )

    velocity_lines = [
        "PLANET SPEEDS",
        "----------------------"
    ]

    for body in planet_names:

        x = positions[body]["x"]
        y = positions[body]["y"]
        z = positions[body]["z"]

        # Move the planet to its current position.
        planet_markers[body].set_data_3d(
            [x[frame]],
            [y[frame]],
            [z[frame]]
        )

        # Draw only the path travelled up to this frame.
        # No future positions are included.
        trail_lines[body].set_data_3d(
            x[:frame + 1],
            y[:frame + 1],
            z[:frame + 1]
        )

        # Update instantaneous speed.
        if speeds[body] is not None:

            speed = speeds[body][frame]

            velocity_lines.append(
                f"{body}: {speed:.3e} m/s"
            )

        else:

            velocity_lines.append(
                f"{body}: unavailable"
            )

    time_text.set_text(
        f"Time = {time_values[frame]:.5f} {time_unit}"
    )

    velocity_text.set_text("\n".join(velocity_lines))

    return (
        [star_marker, time_text, velocity_text]
        + list(planet_markers.values())
        + list(trail_lines.values())
    )

# ============================================================
# 10. CREATE THE ANIMATION
# ============================================================

# Reduce this number for a faster test run.
number_of_frames = min(len(df), 1500)

frame_indices = np.linspace(
    0,
    len(df) - 1,
    number_of_frames,
    dtype=int
)

animation = FuncAnimation(
    fig,
    update,
    frames=frame_indices,
    interval=30,
    blit=False,
    cache_frame_data=False
)

# ============================================================
# 11. SAVE AS MP4
# ============================================================

output_path = (
    Path("/Users/ankitpanda/Desktop")
    / "TRAPPIST1_3D_progressive_trails_velocity.mp4"
)

try:

    writer = FFMpegWriter(
        fps=30,
        bitrate=2400
    )

    animation.save(
        str(output_path),
        writer=writer,
        dpi=80
    )

    print("\nAnimation saved successfully:")
    print(output_path)

except (FileNotFoundError, RuntimeError) as error:

    print("\nCould not save the MP4.")
    print("Check that FFmpeg is installed and accessible.")
    print(error)

plt.show()
#########################################################
#########################################################
#########################################################
#########################################################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from pathlib import Path

# ============================================================
# 1. READ CSV FILE
# ============================================================

file_path = (
    "/Users/ankitpanda/Desktop/"
    "trappist_model_data/TRAPPIST1_simulation_new.csv"
)

df = pd.read_csv(file_path)

if len(df) < 2:
    raise ValueError("The CSV must contain at least two time steps.")

print("Number of time steps:", len(df))


# ============================================================
# 2. IDENTIFY STAR AND COM POSITION COLUMNS
# ============================================================

# Find the central star's X position in AU.
star_x_candidates = [
    col for col in df.columns
    if col.startswith("TRAPPIST")
    and col.endswith("_x_AU")
    and not any(
        col.endswith(f" {planet}_x_AU")
        for planet in "bcdefgh"
    )
]

if not star_x_candidates:
    raise ValueError(
        "Could not find the star's X-position column in AU."
    )

star_prefix = star_x_candidates[0][:-len("_x_AU")]

star_x_col = star_prefix + "_x_AU"
star_y_col = star_prefix + "_y_AU"
star_z_col = star_prefix + "_z_AU"

com_x_col = "COM_x_AU"
com_y_col = "COM_y_AU"
com_z_col = "COM_z_AU"

required_columns = [
    star_x_col, star_y_col, star_z_col,
    com_x_col, com_y_col, com_z_col
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

print("Star columns:", star_x_col, star_y_col, star_z_col)


# ============================================================
# 3. CALCULATE STAR POSITION RELATIVE TO COM
# ============================================================

x = df[star_x_col].to_numpy(dtype=float) - \
    df[com_x_col].to_numpy(dtype=float)

y = df[star_y_col].to_numpy(dtype=float) - \
    df[com_y_col].to_numpy(dtype=float)

z = df[star_z_col].to_numpy(dtype=float) - \
    df[com_z_col].to_numpy(dtype=float)

# Star-COM separation at each time step
separation = np.sqrt(x**2 + y**2 + z**2)

# Time in years
if "Time_years" in df.columns:
    time_years = df["Time_years"].to_numpy(dtype=float)
elif "Time_seconds" in df.columns:
    time_years = (
        df["Time_seconds"].to_numpy(dtype=float)
        / (365.25 * 24 * 3600)
    )
else:
    time_years = np.arange(len(df), dtype=float)

print(f"Maximum star-COM separation: {np.max(separation):.6e} AU")


# ============================================================
# 4. SELECT ANIMATION FRAMES
# ============================================================

number_of_frames = min(len(df), 1200)

frame_indices = np.linspace(
    0, len(df) - 1, number_of_frames, dtype=int
)

# Axis limits based only on the star's relative trajectory
max_range = max(
    np.max(np.abs(x)),
    np.max(np.abs(y)),
    np.max(np.abs(z)),
    1e-12
)

limit = 1.15 * max_range


# ============================================================
# 5. CREATE THE 3D FIGURE
# ============================================================

fig = plt.figure(figsize=(12, 10), facecolor="black")
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("black")

# Dark background and subtle grid
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.set_pane_color((0.015, 0.02, 0.05, 1))
    axis._axinfo["grid"]["color"] = (0.35, 0.55, 0.75, 0.25)
    axis._axinfo["grid"]["linewidth"] = 0.6

ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
ax.set_zlim(-limit, limit)
ax.set_box_aspect((1, 1, 1))

ax.set_xlabel("X relative to COM (AU)", color="white", labelpad=10)
ax.set_ylabel("Y relative to COM (AU)", color="white", labelpad=10)
ax.set_zlabel("Z relative to COM (AU)", color="white", labelpad=10)
ax.tick_params(colors="white", labelsize=9)

fig.suptitle(
    "TRAPPIST-1: 3D Stellar Orbit Around the COM",
    color="white",
    fontsize=16,
    fontweight="bold",
    y=0.94
)


# ============================================================
# 6. MARK THE COM AND CREATE THE STAR TRAJECTORY
# ============================================================

# COM is the origin of the relative coordinate system
ax.scatter(
    [0], [0], [0],
    color="cyan",
    marker="x",
    s=100,
    linewidths=3,
    label="Centre of Mass"
)

# Progressive trajectory line; initially empty
trail, = ax.plot(
    [], [], [],
    color="#39D5FF",
    linewidth=2.2,
    alpha=0.95,
    label="Star trajectory"
)

# Moving star marker
star, = ax.plot(
    [], [], [],
    marker="*",
    markersize=15,
    color="gold",
    markeredgecolor="white",
    markeredgewidth=0.7,
    linestyle="None",
    label="TRAPPIST-1"
)

# Glow effect around the star
glow, = ax.plot(
    [], [], [],
    marker="o",
    markersize=25,
    color="orange",
    alpha=0.18,
    linestyle="None"
)

# Time and separation display in the figure margin
time_text = fig.text(
    0.08, 0.87, "",
    color="white",
    fontsize=12,
    fontweight="bold"
)

distance_text = fig.text(
    0.08, 0.83, "",
    color="#39D5FF",
    fontsize=10
)

ax.legend(
    loc="upper right",
    facecolor="#111827",
    edgecolor="#44516B",
    labelcolor="white",
    fontsize=9
)

fig.subplots_adjust(
    left=0.03,
    right=0.97,
    bottom=0.05,
    top=0.79
)


# ============================================================
# 7. ANIMATION UPDATE FUNCTION
# ============================================================

def update(frame_number):

    # Original CSV row corresponding to this frame
    i = frame_indices[frame_number]

    # Draw only the portion of the orbit already travelled
    trail.set_data_3d(
        x[:i + 1],
        y[:i + 1],
        z[:i + 1]
    )

    # Update the star's current position
    star.set_data_3d(
        [x[i]], [y[i]], [z[i]]
    )

    # Update the glow around the star
    glow.set_data_3d(
        [x[i]], [y[i]], [z[i]]
    )

    # Update time and separation information
    time_text.set_text(
        f"Elapsed time: {time_years[i]:.5f} years"
    )

    distance_text.set_text(
        f"Star-COM separation: {separation[i]:.6e} AU"
    )

    # Slowly rotate the camera to show the 3D trajectory
    ax.view_init(
        elev=25,
        azim=35 + 0.06 * frame_number
    )

    return trail, star, glow, time_text, distance_text


# ============================================================
# 8. CREATE AND SAVE ANIMATION
# ============================================================

animation = FuncAnimation(
    fig,
    update,
    frames=number_of_frames,
    interval=35,
    blit=False,
    repeat=False,
    cache_frame_data=False
)

output_path = Path(
    "/Users/ankitpanda/Desktop/"
    "TRAPPIST1_Star_Orbit_Around_COM1.mp4"
)

try:
    writer = FFMpegWriter(
        fps=25,
        bitrate=2400,
        metadata={"title": "TRAPPIST-1 Stellar Orbit Around COM"}
    )

    animation.save(
        str(output_path),
        writer=writer,
        dpi=100
    )

    print("\nAnimation saved to:")
    print(output_path)

except (FileNotFoundError, RuntimeError) as error:
    print("MP4 export failed:", error)
    print("Trying GIF export instead.")

    gif_path = output_path.with_suffix(".gif")

    animation.save(
        str(gif_path),
        writer=PillowWriter(fps=20),
        dpi=75
    )

    print("GIF saved to:")
    print(gif_path)

plt.close(fig)
