
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## ============================================================
## Read CSV file
## ============================================================

df = pd.read_csv(
    "/Users/ankitpanda/Downloads/solar_system_data_real165yr.csv"
)

AU = 1.495978707e11

## ============================================================
## Extract Mercury coordinates in AU
## Exact column names from the main CSV
## ============================================================

mercury_x = df["Mercury_x_m"].to_numpy() / AU
mercury_y = df["Mercury_y_m"].to_numpy() / AU
mercury_z = df["Mercury_z_m"].to_numpy() / AU

## ============================================================
## 3D Plot of Mercury Orbit
## ============================================================

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111, projection="3d")

# Plot complete Mercury trajectory
ax.plot(
    mercury_x,
    mercury_y,
    mercury_z,
    label="Mercury Orbit",
    linewidth=1
)

# Mark initial position
ax.scatter(
    mercury_x[0],
    mercury_y[0],
    mercury_z[0],
    s=40,color='k',
    label="Initial Position"
)

# Mark final position
ax.scatter(
    mercury_x[-1],
    mercury_y[-1],
    mercury_z[-1],
    s=40,color='r',
    label="Final Position"
)

ax.set_title("Mercury Orbit in 3D")

ax.set_xlabel("X position (AU)")
ax.set_ylabel("Y position (AU)")
ax.set_zlabel("Z position (AU)")

ax.legend()

plt.tight_layout()

#plt.savefig("Mercury_Orbit_3D.pdf",bbox_inches="tight")

plt.show()
