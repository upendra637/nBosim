import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ============================================================
# Constants
# ============================================================

G = 6.67430e-11              # m^3 kg^-1 s^-2
M_sun = 1.989e30             # kg
M_earth = 5.9722e24           # kg
AU = 1.495978707e11           # m
DAY = 24 * 60 * 60
YEAR = 365.25 * DAY


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

                if r_mag == 0:
                    raise ValueError("Two bodies have the same position.")

                accelerations[i] += G * masses[j] * r / r_mag**3

    return accelerations


# ============================================================
# Computing the energy of the system
# ============================================================

def calculate_energy(masses, positions, velocities):
    n = len(masses)
    kinetic_energy = 0.0

    for i in range(n):
        kinetic_energy += (
            0.5 * masses[i] * np.linalg.norm(velocities[i])**2
        )

    potential_energy = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            r_mag = np.linalg.norm(positions[i] - positions[j])

            if r_mag == 0:
                raise ValueError("Two bodies have the same position.")

            potential_energy -= G * masses[i] * masses[j] / r_mag

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
    k1_r, k1_v = derivatives(positions, velocities, masses)

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

    new_positions = positions + (dt / 6.0) * (
        k1_r + 2 * k2_r + 2 * k3_r + k4_r
    )

    new_velocities = velocities + (dt / 6.0) * (
        k1_v + 2 * k2_v + 2 * k3_v + k4_v
    )

    return new_positions, new_velocities


# ============================================================
# TRAPPIST-1 system data
# ============================================================

names = [
    "TRAPPIST-1",
    "TRAPPIST-1 b",
    "TRAPPIST-1 c",
    "TRAPPIST-1 d",
    "TRAPPIST-1 e",
    "TRAPPIST-1 f",
    "TRAPPIST-1 g",
    "TRAPPIST-1 h"
]


# ============================================================
# Masses
# Keep the representative masses from the original code
# ============================================================

masses = np.array([
    0.0898 * M_sun,       # TRAPPIST-1
    1.374 * M_earth,      # TRAPPIST-1 b
    1.308 * M_earth,      # TRAPPIST-1 c
    0.388 * M_earth,      # TRAPPIST-1 d
    0.692 * M_earth,      # TRAPPIST-1 e
    1.039 * M_earth,      # TRAPPIST-1 f
    1.321 * M_earth,      # TRAPPIST-1 g
    0.326 * M_earth       # TRAPPIST-1 h
])


# ============================================================
# Orbital parameters from the user's attached table
# ============================================================

# Semi-major axes in AU: b, c, d, e, f, g, h
semi_major_axes_AU = np.array([
    0.01154,
    0.01580,
    0.02227,
    0.02925,
    0.03849,
    0.04683,
    0.06189
])

# Eccentricities from the attached table
eccentricities = np.array([
    0.00622,
    0.00654,
    0.00837,
    0.00510,
    0.01007,
    0.00208,
    0.00567
])

# Inclinations in degrees from the attached table
inclinations_deg = np.array([
    89.728,
    89.778,
    89.896,
    89.793,
    89.740,
    89.742,
    89.805
])
inclinations_rad = np.radians(inclinations_deg)

# Orbital periods in Earth years, converted from the values in the original code
orbital_periods_years = np.array([
    0.004136416153,  # b
    0.006630902122,  # c
    0.01108615743,   # d
    0.01670366324,   # e
    0.02520887064,   # f
    0.03381915400,   # g
    0.05139730595    # h
])


# ============================================================
# Initial phase angles
# Simplified circular-orbit method, as in the original code
# ============================================================

# Initial phase angles in radians
# These specify each planet's starting position on its orbit.
theta = np.array([
    0.0,
    0.7,
    1.5,
    2.2,
    3.0,
    3.8,
    4.6
])


# ============================================================
# Initial positions and velocities
# Circular orbits with orbital inclinations
# ============================================================

n = len(masses)
positions = np.zeros((n, 3))
velocities = np.zeros((n, 3))

for i in range(1, n):

    # Semi-major axis in metres; for circular orbits, a is the radius
    a = semi_major_axes_AU[i - 1] * AU

    # Inclination in radians
    inc = inclinations_rad[i - 1]

    # Gravitational parameter for the star-planet pair
    mu = G * (masses[0] + masses[i])

    # Circular orbital speed
    speed = np.sqrt(mu / a)

    # Initial position in 3D, using the original sine/cosine approach
    positions[i] = np.array([
        a * np.cos(theta[i - 1]),
        a * np.sin(theta[i - 1]) * np.cos(inc),
        a * np.sin(theta[i - 1]) * np.sin(inc)
    ])

    # Initial velocity for a circular orbit, tilted by inclination
    velocities[i] = np.array([
        -speed * np.sin(theta[i - 1]),
        speed * np.cos(theta[i - 1]) * np.cos(inc),
        speed * np.cos(theta[i - 1]) * np.sin(inc)
    ])

# Star starts at the origin before the COM transformation
positions[0] = np.zeros(3)
velocities[0] = np.zeros(3)

print("\nTRAPPIST-1 approximate initial conditions")
print("--------------------------------------------")

for i in range(n):
    print(f"\n{names[i]}")
    print("Position (AU) =", positions[i] / AU)
    print("Velocity (m/s) =", velocities[i])

    if i > 0:
        print("Semi-major axis / circular radius (AU) =",
              semi_major_axes_AU[i - 1])
        print("Inclination (degrees) =", inclinations_deg[i - 1])
        print("Initial phase angle (degrees) =",
              np.degrees(theta[i - 1]))


# ============================================================
# Transform to Centre of Mass frame
# ============================================================

total_mass = np.sum(masses)

com_position = (
    np.sum(masses[:, np.newaxis] * positions, axis=0) / total_mass
)

com_velocity = (
    np.sum(masses[:, np.newaxis] * velocities, axis=0) / total_mass
)

positions = positions - com_position
velocities = velocities - com_velocity

print("\nCentre of Mass frame")
print("--------------------------------------------")

print(
    "Initial COM position (m) =",
    np.sum(masses[:, np.newaxis] * positions, axis=0) / total_mass
)

print(
    "Initial COM velocity (m/s) =",
    np.sum(masses[:, np.newaxis] * velocities, axis=0) / total_mass
)


# ============================================================
# Orbital periods and planet-specific time steps
# ============================================================

steps_per_orbit = 1000
orbital_periods = np.zeros(n)

for i in range(1, n):
    orbital_periods[i] = orbital_periods_years[i - 1] * YEAR

dt_planets = np.zeros(n)

for i in range(1, n):
    dt_planets[i] = orbital_periods[i] / steps_per_orbit

print("\nPlanetary timestep information")
print("--------------------------------------------")

for i in range(1, n):
    period_years = orbital_periods[i] / YEAR
    timestep_days = dt_planets[i] / DAY

    print(
        f"{names[i]:14s} : "
        f"T = {period_years:8.5f} years   "
        f"dt = {timestep_days:10.6f} days"
    )

dt = np.min(dt_planets[1:])

print("\n--------------------------------------------")
print(f"Global timestep = {dt / YEAR:.10f} Earth years")


# ============================================================
# Simulation parameters
# ============================================================

years = 1
total_time = years * YEAR
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

com_position_history = np.zeros((steps, 3))
com_velocity_history = np.zeros((steps, 3))


# ============================================================
# Simulation
# ============================================================

for step in range(steps):
    # Store positions and velocities
    trajectories[step] = positions
    velocities_history[step] = velocities

    # Calculate centre of mass position and velocity
    com_position_history[step] = (
        np.sum(masses[:, np.newaxis] * positions, axis=0) / total_mass
    )

    com_velocity_history[step] = (
        np.sum(masses[:, np.newaxis] * velocities, axis=0) / total_mass
    )

    # Calculate energy
    K, U, E = calculate_energy(masses, positions, velocities)

    kinetic_energy_history[step] = K
    potential_energy_history[step] = U
    total_energy_history[step] = E

    # Advance by one RK4 step
    new_positions, new_velocities = rk4_step(
        positions,
        velocities,
        masses,
        dt
    )

    positions = new_positions
    velocities = new_velocities


# ============================================================
# Energy conservation
# ============================================================

initial_energy = total_energy_history[0]

relative_energy_error = (
    np.abs(total_energy_history - initial_energy) / np.abs(initial_energy)
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
# Centre of Mass conservation
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
# 3D Plot
# ============================================================

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

# Plot planetary trajectories
for i in range(1, n):
    ax.plot(
        trajectories_AU[:, i, 0],
        trajectories_AU[:, i, 1],
        trajectories_AU[:, i, 2],
        label=names[i]
    )

# Plot the star
ax.plot(
    trajectories_AU[:, 0, 0],
    trajectories_AU[:, 0, 1],
    trajectories_AU[:, 0, 2],
    label="TRAPPIST-1"
)

# COM is at the origin in the ideal COM frame
ax.scatter(0, 0, 0, s=20, label="Centre of Mass")

ax.set_title(
    "3D N-Body TRAPPIST-1 Simulation\n"
    "Circular Orbits with Inclination Included",
    fontsize=14
)

ax.set_xlabel("X position (AU)")
ax.set_ylabel("Y position (AU)")
ax.set_zlabel("Z position (AU)")
ax.legend()
plt.tight_layout()
plt.show()


# ============================================================
# Energy conservation plot
# ============================================================

time_years = np.arange(steps) * dt / YEAR

plt.figure(figsize=(10, 6))
plt.plot(time_years, kinetic_energy_history, label="Kinetic Energy")
plt.plot(time_years, potential_energy_history, label="Potential Energy")
plt.plot(time_years, total_energy_history, label="Total Energy")
plt.xlabel("Time (years)")
plt.ylabel("Energy (J)")
plt.title("Energy Conservation in TRAPPIST-1 N-Body Simulation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# Relative Energy Error
# ============================================================

plt.figure(figsize=(10, 6))
plt.plot(time_years, relative_energy_error)
plt.xlabel("Time (years)")
plt.ylabel("Relative Energy Error")
plt.title("Relative Energy Error")
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# Centre of Mass Position
# ============================================================

plt.figure(figsize=(10, 6))
plt.plot(time_years, com_position_history[:, 0] / AU, label="COM x")
plt.plot(time_years, com_position_history[:, 1] / AU, label="COM y")
plt.plot(time_years, com_position_history[:, 2] / AU, label="COM z")
plt.xlabel("Time (years)")
plt.ylabel("COM Position (AU)")
plt.title("Centre of Mass Position vs Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
