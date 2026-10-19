
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. Constants
# ============================================================

AU = 1.495978707e11  # Astronomical unit in metres

# ============================================================
# 2. Read the CSV file
# ============================================================

file_path = "/Users/ankitpanda/Downloads/solar_system_data_real165yr.csv"

df = pd.read_csv(file_path)

# ============================================================
# 3. Extract Sun and Jupiter coordinates in the COM frame
# ============================================================

sun_x = df["Sun_x_m"].to_numpy()
sun_y = df["Sun_y_m"].to_numpy()
sun_z = df["Sun_z_m"].to_numpy()

jupiter_x = df["Jupiter_x_m"].to_numpy()
jupiter_y = df["Jupiter_y_m"].to_numpy()
jupiter_z = df["Jupiter_z_m"].to_numpy()

# ============================================================
# 4. Convert coordinates from metres to AU
# ============================================================

sun_x_AU = sun_x / AU
sun_y_AU = sun_y / AU
sun_z_AU = sun_z / AU

jupiter_x_AU = jupiter_x / AU
jupiter_y_AU = jupiter_y / AU
jupiter_z_AU = jupiter_z / AU

# ============================================================
# 5. Plot Sun and Jupiter orbits in the COM frame
# ============================================================

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111, projection="3d")

# Sun orbit
ax.plot(
    sun_x_AU,
    sun_y_AU,
    sun_z_AU,
    label="Sun",
    color="orange"
)

# Jupiter orbit
ax.plot(
    jupiter_x_AU,
    jupiter_y_AU,
    jupiter_z_AU,
    label="Jupiter",
    color="blue"
)

# Centre of Mass
ax.scatter(
    0, 0, 0,
    color="black",
    marker="x",
    s=60,
    label="COM"
)

ax.set_xlabel("X (AU)")
ax.set_ylabel("Y (AU)")
ax.set_zlabel("Z (AU)")

ax.set_title("Sun and Jupiter Orbits in the COM Frame")

ax.legend()

plt.tight_layout()

plt.savefig(
    "Sun_Jupiter_orbits_COM.pdf",
    bbox_inches="tight"
)

plt.show()
