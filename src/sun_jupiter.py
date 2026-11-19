
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import requests


# ============================================================
# Constants
# ============================================================

G = 6.67430e-11
M_sun = 1.989e30
AU = 1.495978707e11
DAY = 24 * 60 * 60


# ============================================================
# Computing acceleration
# ============================================================

def calculate_acceleration(masses, positions):

    accelerations = np.zeros_like(positions)
    n = len(masses)

    for i in range(n):
        for j in range(n):
            if i != j:
                r = positions[j] - positions[i]
                r_mag = np.linalg.norm(r)

                accelerations[i] += (
                    G * masses[j] * r / r_mag**3
                )

    return accelerations


# ============================================================
# Computing the energy of the system
# ============================================================

def calculate_energy(masses, positions, velocities):

    n = len(masses)
    kinetic_energy = 0

    for i in range(n):
        kinetic_energy += (
            0.5 * masses[i]
            * np.linalg.norm(velocities[i])**2
        )

    potential_energy = 0

    for i in range(n):
        for j in range(i + 1, n):

            r_mag = np.linalg.norm(
                positions[i] - positions[j]
            )

            potential_energy -= (
                G * masses[i] * masses[j] / r_mag
            )

    total_energy = kinetic_energy + potential_energy

    return kinetic_energy, potential_energy, total_energy


# ============================================================
# Equations of motion
# ============================================================

def derivatives(positions, velocities, masses):

    dr_dt = velocities
    dv_dt = calculate_acceleration(masses, positions)

    return dr_dt, dv_dt


# ============================================================
# RK4 method
# ============================================================

def rk4_step(positions, velocities, masses, dt):

    # k1
    k1_r, k1_v = derivatives(
        positions, velocities, masses
    )

    # k2
    k2_r, k2_v = derivatives(
        positions + 0.5 * dt * k1_r,
        velocities + 0.5 * dt * k1_v,
        masses
    )

    # k3
    k3_r, k3_v = derivatives(
        positions + 0.5 * dt * k2_r,
        velocities + 0.5 * dt * k2_v,
        masses
    )

    # k4
    k4_r, k4_v = derivatives(
        positions + dt * k3_r,
        velocities + dt * k3_v,
        masses
    )

    new_positions = (
        positions
        + (dt / 6)
        * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
    )

    new_velocities = (
        velocities
        + (dt / 6)
        * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
    )

    return new_positions, new_velocities


# ============================================================
# Solar System Data: Sun and Jupiter only
# ============================================================

names = [
    "Sun",
    "Jupiter"
]


# ============================================================
# Masses
# ============================================================

masses_solar = np.array([
    1.000000,          # Sun
    1.898e27 / M_sun   # Jupiter
])

# Convert masses to kg
masses = masses_solar * M_sun


# ============================================================
# NASA / JPL Horizons body IDs
# ============================================================

body_ids = [
    "10",    # Sun
    "599"    # Jupiter
]


# ============================================================
# NASA/JPL Horizons epoch
# ============================================================

epoch = "2026-10-05"

print("\nNASA/JPL Horizons initial conditions")
print("--------------------------------------------")
print("Epoch =", epoch)


# ============================================================
# Function to obtain state vector from JPL Horizons
# ============================================================

def get_horizons_state(body_id, epoch):

    url = "https://ssd.jpl.nasa.gov/api/horizons.api"

    params = {
        "format": "text",
        "COMMAND": f"'{body_id}'",
        "OBJ_DATA": "'NO'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'VECTORS'",

        # Solar System barycentre
        "CENTER": "'@0'",

        "START_TIME": f"'{epoch}'",
        "STOP_TIME": f"'{epoch} 00:01'",
        "STEP_SIZE": "'1 m'",

        # Position + velocity
        "VEC_TABLE": "'2'",

        # AU and AU/day
        "OUT_UNITS": "'AU-D'",

        # ICRF reference frame
        "REF_SYSTEM": "'ICRF'",

        # Equatorial reference plane
        "REF_PLANE": "'FRAME'",

        # Geometric state
        "VEC_CORR": "'NONE'",
        "CSV_FORMAT": "'YES'"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    text = response.text

    # Find state-vector data between $$SOE and $$EOE
    start = text.find("$$SOE")
    end = text.find("$$EOE")

    if start == -1 or end == -1:
        print(text)
        raise RuntimeError(
            "Could not find Horizons state-vector data."
        )

    data = text[
        start + len("$$SOE"):end
    ].strip()

    # Horizons CSV line
    line = data.splitlines()[0]
    values = line.split(",")

    # X, Y, Z in AU
    # VX, VY, VZ in AU/day
    x = float(values[2])
    y = float(values[3])
    z = float(values[4])

    vx = float(values[5])
    vy = float(values[6])
    vz = float(values[7])

    return (
        np.array([x, y, z]),
        np.array([vx, vy, vz])
    )


# ============================================================
# Get NASA/JPL initial positions and velocities
# ============================================================

n = len(masses)

positions = np.zeros((n, 3))
velocities = np.zeros((n, 3))

for i in range(n):

    position_AU, velocity_AU_day = get_horizons_state(
        body_ids[i], epoch
    )

    # Convert position: AU -> m
    positions[i] = position_AU * AU

    # Convert velocity: AU/day -> m/s
    velocities[i] = velocity_AU_day * AU / DAY

    print(
        f"{names[i]:8s} : r = {positions[i] / AU} AU"
    )

    print(
        f"{'':8s}   v = {velocities[i]} m/s"
    )


# ============================================================
# Transform to Centre of Mass frame
# ============================================================

total_mass = np.sum(masses)

# Centre of mass position
com_position = (
    np.sum(masses[:, np.newaxis] * positions, axis=0)
    / total_mass
)

# Centre of mass velocity
com_velocity = (
    np.sum(masses[:, np.newaxis] * velocities, axis=0)
    / total_mass
)

# Shift positions to COM frame
positions = positions - com_position

# Shift velocities to COM frame
velocities = velocities - com_velocity

print("\nCentre of Mass frame")
print("--------------------------------------------")

print(
    "Initial COM position (m) =",
    np.sum(
        masses[:, np.newaxis] * positions,
        axis=0
    ) / total_mass
)

print(
    "Initial COM velocity (m/s) =",
    np.sum(
        masses[:, np.newaxis] * velocities,
        axis=0
    ) / total_mass
)


# ============================================================
# Orbital periods
# ============================================================

steps_per_orbit = 1000
orbital_periods = np.zeros(n)

for i in range(1, n):

    orbital_periods[i] = (
        2 * np.pi
        * np.sqrt(
            np.linalg.norm(positions[i])**3
            / (G * masses[0])
        )
    )


# ============================================================
# Planet-specific time steps
# ============================================================

dt_planets = np.zeros(n)

for i in range(1, n):

    dt_planets[i] = (
        orbital_periods[i] / steps_per_orbit
    )


# Display orbital periods
print("\nPlanetary timestep information")
print("--------------------------------------------")

for i in range(1, n):

    period_years = (
        orbital_periods[i] / (365.25 * DAY)
    )

    timestep_days = dt_planets[i] / DAY

    print(
        f"{names[i]:8s} : "
        f"T = {period_years:8.3f} years   "
        f"dt = {timestep_days:8.4f} days"
    )


# ============================================================
# Global timestep
# ============================================================

dt = np.min(dt_planets[1:])

print("\n--------------------------------------------")
print(f"Global timestep = {dt / DAY:.4f} days")


# ============================================================
# Simulation parameters
# ============================================================

years = 400

total_time = years * 365.25 * DAY

steps = int(total_time / dt)

print(f"\nTotal simulation time = {years} years")
print(f"Total integration steps = {steps}")


# ============================================================
# Storage arrays
# ============================================================

trajectories = np.zeros((steps, n, 3))
velocities_history = np.zeros((steps, n, 3))

kinetic_energy_history = np.zeros(steps)
potential_energy_history = np.zeros(steps)
total_energy_history = np.zeros(steps)

# Centre of mass history
com_position_history = np.zeros((steps, 3))
com_velocity_history = np.zeros((steps, 3))


# ============================================================
# Simulation
# ============================================================

for step in range(steps):

    # Store positions and velocities
    trajectories[step] = positions
    velocities_history[step] = velocities

    # Calculate Centre of Mass
    com_position_history[step] = (
        np.sum(
            masses[:, np.newaxis] * positions,
            axis=0
        ) / total_mass
    )

    com_velocity_history[step] = (
        np.sum(
            masses[:, np.newaxis] * velocities,
            axis=0
        ) / total_mass
    )

    # Calculate energy
    K, U, E = calculate_energy(
        masses, positions, velocities
    )

    kinetic_energy_history[step] = K
    potential_energy_history[step] = U
    total_energy_history[step] = E

    # RK4 step
    new_positions, new_velocities = rk4_step(
        positions, velocities, masses, dt
    )

    positions = new_positions
    velocities = new_velocities


# ============================================================
# Energy Conservation
# ============================================================

initial_energy = total_energy_history[0]

relative_energy_error = (
    np.abs(total_energy_history - initial_energy)
    / np.abs(initial_energy)
)

print("\nEnergy conservation")
print("--------------------------------------------")

print(f"Initial total energy = {initial_energy:.6e} J")
print(f"Final total energy   = {total_energy_history[-1]:.6e} J")

print(
    "Maximum relative energy error = "
    f"{np.max(np.abs(relative_energy_error)):.6e}"
)

print(
    "Final relative energy error = "
    f"{relative_energy_error[-1]:.6e}"
)


# ============================================================
# Centre of Mass Conservation
# ============================================================

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


# ============================================================
# Convert trajectories to AU
# ============================================================

trajectories_AU = trajectories / AU


# ============================================================
# 3D Plot: Sun and Jupiter
# ============================================================

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot Jupiter trajectory
ax.plot(
    trajectories_AU[:, 1, 0],
    trajectories_AU[:, 1, 1],
    trajectories_AU[:, 1, 2],
    label="Jupiter"
)

# Plot Sun trajectory
ax.plot(
    trajectories_AU[:, 0, 0],
    trajectories_AU[:, 0, 1],
    trajectories_AU[:, 0, 2],
    label="Sun"
)

# Plot COM
ax.scatter(
    0, 0, 0,
    s=5,
    label="Centre of Mass"
)

ax.set_title(
    "Sun-Jupiter Two-Body Simulation\n"
    "NASA/JPL Horizons Initial Conditions",
    fontsize=14
)

ax.set_xlabel("X position (AU)")
ax.set_ylabel("Y position (AU)")
ax.set_zlabel("Z position (AU)")

ax.legend()

plt.savefig(
    "/Users/ankitpanda/Desktop/orbits._sun_jupiter.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# Energy Conservation Plot
# ============================================================

time_years = (
    np.arange(steps) * dt / (365.25 * DAY)
)

plt.figure(figsize=(10, 6))

plt.plot(
    time_years,
    kinetic_energy_history,
    label="Kinetic Energy"
)

plt.plot(
    time_years,
    potential_energy_history,
    label="Potential Energy"
)

plt.plot(
    time_years,
    total_energy_history,
    label="Total Energy"
)

plt.xlabel("Time (years)")
plt.ylabel("Energy (J)")

plt.title(
    "Energy Conservation in Sun-Jupiter Simulation"
)

plt.legend()
plt.grid(True)

plt.savefig(
    "/Users/ankitpanda/Desktop/energy_conservation_sun_jupiter.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# Relative Energy Error
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    time_years,
    relative_energy_error
)

plt.xlabel("Time (years)")
plt.ylabel("Relative Energy Error")

plt.title("Relative Energy Error")

plt.grid(True)

plt.savefig(
    "/Users/ankitpanda/Desktop/relative_energy_error_sun_jupiter.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# Centre of Mass Position
# ============================================================

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

plt.savefig(
    "/Users/ankitpanda/Desktop/COM_position_sun_jupiter.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
