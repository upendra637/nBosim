
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## ============================================================
## Read Sun trajectory CSV file
## ============================================================

df = pd.read_csv(
    "/Users/ankitpanda/Downloads/sun_trajectory_COM.csv"
)

# Display column names
print(df.columns.tolist())

# Display first five rows
print(df.head())


## ============================================================
## Extract Sun coordinates relative to COM
## ============================================================

sun_x = df["X_relative_to_COM_AU"].to_numpy()
sun_y = df["Y_relative_to_COM_AU"].to_numpy()
sun_z = df["Z_relative_to_COM_AU"].to_numpy()


## ============================================================
## 3D Plot of Sun Trajectory in COM Frame
## ============================================================

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111, projection="3d")

# Plot Sun trajectory
ax.plot(
    sun_x,
    sun_y,
    sun_z,
    label="Sun Trajectory",
    linewidth=1
)

# Mark initial position
ax.scatter(
    sun_x[0],
    sun_y[0],
    sun_z[0],
    label="Initial Position",
    s=40
)

# Mark final position
ax.scatter(
    sun_x[-1],
    sun_y[-1],
    sun_z[-1],
    label="Final Position",
    s=40
)

# Mark Centre of Mass
ax.scatter(
    0, 0, 0,
    label="Centre of Mass",
    s=50
)

# Labels and title
ax.set_title(
    "Sun Trajectory Relative to the Centre of Mass",
    fontsize=13
)

ax.set_xlabel("X position (AU)")
ax.set_ylabel("Y position (AU)")
ax.set_zlabel("Z position (AU)")

ax.legend()

plt.tight_layout()

# Save figure as PDF
plt.savefig(
    "sun_trajectory_COM.pdf",
    bbox_inches="tight"
)

plt.show()
##########################################
#########################################

import pandas as pd
import matplotlib.pyplot as plt

## ============================================================
## Read CSV file
## ============================================================

df = pd.read_csv(
    "/Users/ankitpanda/Desktop/sun_COM_separation.csv"
)

# Display column names
print(df.columns.tolist())

## ============================================================
## Extract data
## ============================================================

time_years = df["Time_years"].to_numpy()

sun_COM_separation_AU = df["Sun_COM_separation_AU"].to_numpy()

## ============================================================
## Plot Sun-COM Separation vs Time
## ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    time_years,
    sun_COM_separation_AU,
    linewidth=1
)

plt.xlabel("Time (years)")
plt.ylabel("Sun-COM Separation (AU)")

plt.title("Sun-COM Separation vs Time")

plt.grid(True)

plt.tight_layout()

# Save plot as PDF
plt.savefig(
    "Sun_COM_Separation_vs_Time.pdf",
    bbox_inches="tight"
)

plt.show()
