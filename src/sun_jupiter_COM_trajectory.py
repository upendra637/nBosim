
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. Read the CSV file
# ============================================================

file_path = "/Users/ankitpanda/Desktop/sun_jupiter_simulation_data.csv"

df = pd.read_csv(file_path)


# ============================================================
# 2. Extract Sun's position with respect to COM
# ============================================================

sun_x = df["Sun_x_AU"]
sun_y = df["Sun_y_AU"]
sun_z = df["Sun_z_AU"]


# ============================================================
# 3. Plot the Sun's 3D trajectory
# ============================================================

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111, projection="3d")

ax.plot(
    sun_x,
    sun_y,
    sun_z,
    label="Sun trajectory"
)

# Mark the initial position
ax.scatter(
    sun_x.iloc[0],
    sun_y.iloc[0],
    sun_z.iloc[0],
    label="Initial position",
    s=40
)

# Mark the centre of mass
ax.scatter(
    0, 0, 0,
    label="Centre of Mass",
    s=50
)

# ============================================================
# 4. Labels and formatting
# ============================================================

ax.set_xlabel("X position (AU)")
ax.set_ylabel("Y position (AU)")
ax.set_zlabel("Z position (AU)")

ax.set_title("3D Trajectory of the Sun with Respect to COM")

ax.legend()

plt.tight_layout()

# Save the plot
plt.savefig(
    "/Users/ankitpanda/Desktop/Sun_trajectory_wrt_COM.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
