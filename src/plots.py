
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## ============================================================
## Read CSV file
## ============================================================

df = pd.read_csv(
    "/Users/ankitpanda/Downloads/solar_system_data_real165yr.csv"
)

print(df.columns.tolist())
print(df.head())


## ============================================================
## Energy Conservation
## ============================================================

initial_energy = df["Total_Energy_J"].iloc[0]

total_energy_history = df["Total_Energy_J"].to_numpy()

relative_energy_error = df["Relative_Energy_Error"].to_numpy()

print("\nEnergy conservation")
print("--------------------------------------------")

print(f"Initial total energy = {initial_energy:.6e} J")

print(f"Final total energy   = {total_energy_history[-1]:.6e} J")

print(
    f"Maximum relative energy error = "
    f"{np.max(np.abs(relative_energy_error)):.6e}"
)

print(
    f"Final relative energy error = "
    f"{relative_energy_error[-1]:.6e}"
)


## ============================================================
## Centre of Mass Conservation
## ============================================================

AU = 1.495978707e11
DAY = 86400

com_position_history = df[
    ["COM_x_m", "COM_y_m", "COM_z_m"]
].to_numpy()

com_velocity_history = df[
    ["COM_vx_m_s", "COM_vy_m_s", "COM_vz_m_s"]
].to_numpy()

print("\nCentre of Mass conservation")
print("--------------------------------------------")

print(
    "Maximum COM position magnitude = ",
    f"{np.max(np.linalg.norm(com_position_history, axis=1)):.6e} m"
)

print(
    "Maximum COM velocity magnitude = ",
    f"{np.max(np.linalg.norm(com_velocity_history, axis=1)):.6e} m/s"
)


## ============================================================
## 3D N-Body Orbits in COM Frame
## ============================================================

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

fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111, projection="3d")

for name in names:

    x = df[f"{name}_x_m"].to_numpy() / AU
    y = df[f"{name}_y_m"].to_numpy() / AU
    z = df[f"{name}_z_m"].to_numpy() / AU

    ax.plot(x, y, z, label=name)

ax.scatter(0, 0, 0, s=20, label="Centre of Mass")

ax.set_title(
    "3D N-Body Solar System Simulation "
    "in Centre of Mass Frame using RK4"
)

ax.set_xlabel("X position (AU)")
ax.set_ylabel("Y position (AU)")
ax.set_zlabel("Z position (AU)")

ax.legend()

plt.tight_layout()
# plt.savefig("orbits.pdf", bbox_inches="tight")
plt.show()


## ============================================================
## Energy Conservation Plot
## ============================================================

time_years = df["Time_years"].to_numpy()

kinetic_energy_history = df["Kinetic_Energy_J"].to_numpy()
potential_energy_history = df["Potential_Energy_J"].to_numpy()
total_energy_history = df["Total_Energy_J"].to_numpy()

plt.figure(figsize=(10, 6))

plt.plot(time_years, kinetic_energy_history, label="Kinetic Energy")
plt.plot(time_years, potential_energy_history, label="Potential Energy")
plt.plot(time_years, total_energy_history, label="Total Energy")

plt.xlabel("Time (years)")
plt.ylabel("Energy (J)")
plt.title("Energy Conservation in N-Body Solar System Simulation")

plt.legend()
plt.grid(True)
plt.tight_layout()

# plt.savefig("energy_conservation.pdf", bbox_inches="tight")
plt.show()


## ============================================================
## Relative Energy Error Plot
## ============================================================

plt.figure(figsize=(10, 6))

plt.plot(time_years, relative_energy_error)

plt.xlabel("Time (years)")
plt.ylabel("Relative Energy Error")
plt.title("Relative Energy Error")

plt.grid(True)
plt.tight_layout()

# plt.savefig("relative_energy_error.pdf", bbox_inches="tight")
plt.show()


## ============================================================
## Centre of Mass Position Plot
## ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    time_years,
    com_position_history[:, 0] / AU,
    label="COM x"
)

plt.plot(
    time_years,
    com_position_history[:, 1] / AU,
    label="COM y"
)

plt.plot(
    time_years,
    com_position_history[:, 2] / AU,
    label="COM z"
)

plt.xlabel("Time (years)")
plt.ylabel("COM Position (AU)")
plt.title("Centre of Mass Position vs Time")

plt.legend()
plt.grid(True)
plt.tight_layout()

# plt.savefig("COM_position.pdf", bbox_inches="tight")
plt.show()
