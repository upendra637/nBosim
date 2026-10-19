
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## ============================================================
## Read the two CSV files
## ============================================================

# Main solar-system simulation data
orbit_df = pd.read_csv(
    "/Users/ankitpanda/Downloads/solar_system_data_real165yr.csv"
)

# Mercury perihelion position data
perihelion_df = pd.read_csv(
    "/Users/ankitpanda/Downloads/Mercury_Perihelia_COM_165_initial.csv"
)

AU = 1.495978707e11

## ============================================================
## Mercury's complete orbit
## ============================================================

# Exact column names from the main simulation CSV
mercury_x = orbit_df["Mercury_x_m"].to_numpy() / AU
mercury_y = orbit_df["Mercury_y_m"].to_numpy() / AU
mercury_z = orbit_df["Mercury_z_m"].to_numpy() / AU

## ============================================================
## Mercury's perihelion positions
## ============================================================

# Exact column names from the perihelion CSV
perihelion_x = perihelion_df["Mercury_x_COM_AU"].to_numpy()
perihelion_y = perihelion_df["Mercury_y_COM_AU"].to_numpy()
perihelion_z = perihelion_df["Mercury_z_COM_AU"].to_numpy()

revolution = perihelion_df["Revolution"].to_numpy()

## ============================================================
## 3D Plot: Mercury Orbit with Perihelion Positions
## ============================================================

fig = plt.figure(figsize=(11, 9))

ax = fig.add_subplot(111, projection="3d")

# Plot the complete Mercury orbit
ax.plot(
    mercury_x,
    mercury_y,
    mercury_z,
    label="Mercury Orbit",
    linewidth=0.8
)

# Mark all perihelion positions
ax.scatter(
    perihelion_x,
    perihelion_y,
    perihelion_z,
    color="red",
    s=18,
    label="Perihelion Positions"
)

# Mark the first perihelion
ax.scatter(
    perihelion_x[0],
    perihelion_y[0],
    perihelion_z[0],
    color="green",
    s=65,
    label="First Perihelion"
)

# Mark the centre of mass
ax.scatter(
    0, 0, 0,
    color="black",
    s=35,
    label="Centre of Mass"
)

ax.set_title(
    "Mercury 3D Orbit with Perihelion Positions",
    fontsize=14
)

ax.set_xlabel("X position (AU)")
ax.set_ylabel("Y position (AU)")
ax.set_zlabel("Z position (AU)")

ax.legend()
plt.tight_layout()

# Save the figure
plt.savefig(
    "Mercury_Orbit_with_Perihelia_3D.pdf",
    bbox_inches="tight"
)

plt.show()
