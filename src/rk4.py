import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

G=6.67430e-11  # gravitational constant in m^3 kg^-1 s^-2
M_sun = 1.989e30             # kg
AU = 1.495978707e11          # m


## computing acceleration
def calculate_acceleration(masses, positions):
    accelerations = np.zeros_like(positions)
    n=len(masses)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                r = positions[j] - positions[i]
                r_mag = np.linalg.norm(r)
                accelerations[i] += G * masses[j] * r / r_mag**3
    
    return accelerations


## Computing the energy of the system
def calculate_energy(masses,positions,velocities):
    n=len(masses)
    
    kinetic_energy=0
    
    for i in range(n):
        kinetic_energy+=0.5*masses[i]*np.linalg.norm(velocities[i])**2
    
    potential_energy=0
    
    for i in range(n):
        for j in range(i+1,n):
            r_mag=np.linalg.norm(positions[i]-positions[j])
            potential_energy-=G*masses[i]*masses[j]/r_mag
    
    total_energy=kinetic_energy+potential_energy
    
    return kinetic_energy,potential_energy,total_energy


## computing the equations of motion
def derivatives(positions,velocities,masses):
    dr_dt = velocities
    dv_dt = calculate_acceleration(masses, positions)
    
    return dr_dt, dv_dt


### implementing the RK4 method
def rk4_step(positions, velocities, masses, dt):
    
    # k1
    k1_r, k1_v = derivatives(
        positions,
        velocities,
        masses
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

    new_positions = positions + (dt / 6) * (
        k1_r + 2 * k2_r + 2 * k3_r + k4_r
    )
    
    new_velocities = velocities + (dt / 6) * (
        k1_v + 2 * k2_v + 2 * k3_v + k4_v
    )
    
    return new_positions, new_velocities


## ============================================================
## Solar System Data
## ============================================================

names=[
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


#masses in solar mass
masses_solar = np.array([
    1.000000,               # Sun
    3.301e23 / M_sun,       # Mercury
    4.867e24 / M_sun,       # Venus
    5.972e24 / M_sun,       # Earth
    6.417e23 / M_sun,       # Mars
    1.898e27 / M_sun,       # Jupiter
    5.683e26 / M_sun,       # Saturn
    8.681e25 / M_sun,       # Uranus
    1.024e26 / M_sun,       # Neptune
    1.303e22 / M_sun        # Pluto
])


# converting masses to kg
masses = masses_solar * M_sun


# semi-major axes in AU
semi_major_axes_AU = np.array([
    0.0,       # Sun
    0.387,     # Mercury
    0.723,     # Venus
    1.000,     # Earth
    1.524,     # Mars
    5.203,     # Jupiter
    9.537,     # Saturn
    19.191,    # Uranus
    30.070,    # Neptune
    39.48      # Pluto
])


# converting semi-major axes to meters
semi_major_axes = semi_major_axes_AU * AU


# Eccentricities of the planets
eccentricities = np.array([
    0.0,       # Sun
    0.2056,    # Mercury
    0.0068,    # Venus
    0.0167,    # Earth
    0.0934,    # Mars
    0.0489,    # Jupiter
    0.0565,    # Saturn
    0.0463,    # Uranus
    0.0095,    # Neptune
    0.2488     # Pluto
])


# Inclinations of the planets in degrees
inclinations = np.array([
    0.0,      # Sun
    7.00,     # Mercury
    3.39,     # Venus
    0.00,     # Earth
    1.85,     # Mars
    1.31,     # Jupiter
    2.49,     # Saturn
    0.77,     # Uranus
    1.77,     # Neptune
    17.16     # Pluto
])


# Convert degrees to radians
inclinations = np.radians(inclinations)


## ============================================================
## Initial positions and velocities
## ============================================================

n=len(masses)

positions = np.zeros((n, 3))
velocities = np.zeros((n, 3))

positions[0]=[0, 0, 0]
velocities[0]=[0, 0, 0]

mu=G * masses[0]


for i in range(1, n):

    r_perihelion = (
        semi_major_axes[i]
        * (1 - eccentricities[i])
    )

    v_perihelion = np.sqrt(
        mu * (
            2/r_perihelion
            - 1/semi_major_axes[i]
        )
    )

    inc=inclinations[i]

    positions[i] = [
        r_perihelion,
        0,
        0
    ]

    velocities[i] = [
        0,
        v_perihelion*np.cos(inc),
        v_perihelion*np.sin(inc)
    ]


## ============================================================
## Transform to Centre of Mass frame
## ============================================================

total_mass = np.sum(masses)


# Centre of mass position
com_position = np.sum(
    masses[:, np.newaxis] * positions,
    axis=0
) / total_mass


# Centre of mass velocity
com_velocity = np.sum(
    masses[:, np.newaxis] * velocities,
    axis=0
) / total_mass


# Shift all positions to COM frame
positions = positions - com_position


# Shift all velocities to COM frame
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


## ============================================================
## Orbital Periods
## ============================================================

# Number of numerical steps per orbital period
steps_per_orbit = 1000


# Orbital periods
orbital_periods = np.zeros(n)


for i in range(1, n):

    # Kepler's third law
    #
    # T = 2*pi*sqrt(a^3 / GM)
    #

    orbital_periods[i] = (
        2 * np.pi
        * np.sqrt(
            semi_major_axes[i]**3
            / (G * masses[0])
        )
    )


# Planet specific time steps
dt_planets = np.zeros(n)

for i in range(1, n):
    dt_planets[i] = (
        orbital_periods[i]
        / steps_per_orbit
    )


# Display orbital periods and time steps
print("\nPlanetary timestep information")
print("--------------------------------------------")

for i in range(1, n):

    period_years = (
        orbital_periods[i]
        / (365.25 * 24 * 3600)
    )

    timestep_days = (
        dt_planets[i]
        / (24 * 3600)
    )

    print(
        f"{names[i]:8s} : "
        f"T = {period_years:8.3f} years   "
        f"dt = {timestep_days:8.4f} days"
    )


# Global time step for the simulation
dt = np.min(dt_planets[1:])


DAY=24*60*60


print("\n--------------------------------------------")

print(
    f"Global timestep = "
    f"{dt / DAY:.4f} days"
)


## ============================================================
## Simulation parameters
## ============================================================

years=165

total_time=years*365.25*DAY

steps=int(total_time/dt)

print(
    f"\nTotal simulation time = {years} years"
)

print(
    f"Total integration steps = {steps}"
)


## ============================================================
## Storage arrays
## ============================================================

trajectories = np.zeros((steps, n, 3))

velocities_history = np.zeros(
    (steps, n, 3)
)

kinetic_energy_history = np.zeros(steps)

potential_energy_history = np.zeros(steps)

total_energy_history = np.zeros(steps)


# Centre of mass history
com_position_history = np.zeros(
    (steps, 3)
)

com_velocity_history = np.zeros(
    (steps, 3)
)


## ============================================================
## Simulation
## ============================================================

for step in range(steps):

    # Store positions and velocities
    trajectories[step] = positions

    velocities_history[step] = velocities


    # --------------------------------------------------------
    # Calculate Centre of Mass
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Calculate energy
    # --------------------------------------------------------

    K, U, E = calculate_energy(
        masses,
        positions,
        velocities
    )

    kinetic_energy_history[step] = K

    potential_energy_history[step] = U

    total_energy_history[step] = E


    # --------------------------------------------------------
    # RK4 step
    # --------------------------------------------------------

    new_positions, new_velocities = rk4_step(
        positions,
        velocities,
        masses,
        dt
    )

    positions, velocities = (
        new_positions,
        new_velocities
    )


## ============================================================
## Energy Conservation
## ============================================================

initial_energy = total_energy_history[0]

relative_energy_error = (
    abs(
        total_energy_history
        - initial_energy
    )
    / abs(initial_energy)
)


print("\nEnergy conservation")
print("--------------------------------------------")

print(
    f"Initial total energy = "
    f"{initial_energy:.6e} J"
)

print(
    f"Final total energy   = "
    f"{total_energy_history[-1]:.6e} J"
)

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
## Convert trajectories to AU
## ============================================================

trajectories_AU = trajectories / AU


## ============================================================
## 3D Plot of the orbits in COM frame
## ============================================================

fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(
    111,
    projection='3d'
)


# Plot planetary trajectories
for i in range(1, n):

    ax.plot(
        trajectories_AU[:, i, 0],
        trajectories_AU[:, i, 1],
        trajectories_AU[:, i, 2],
        label=names[i]
    )


# Plot Sun
ax.plot(
    trajectories_AU[:, 0, 0],
    trajectories_AU[:, 0, 1],
    trajectories_AU[:, 0, 2],
    label='Sun'
)


# Plot COM
ax.scatter(
    0,
    0,
    0,
    s=5,
    label='Centre of Mass'
)


# Labels
ax.set_title(
    '3D N-Body Solar System Simulation '
    'in Centre of Mass Frame using RK4',
    fontsize=14
)

ax.set_xlabel('X position (AU)')

ax.set_ylabel('Y position (AU)')

ax.set_zlabel('Z position (AU)')

ax.legend()

plt.show()
plt.savefig("orbits.pdf")


## ============================================================
## Energy conservation plot
## ============================================================

time_years = (
    np.arange(steps)
    * dt
    / (365.25 * DAY)
)


plt.figure(figsize=(10, 6))

plt.plot(
    time_years,
    kinetic_energy_history,
    label='Kinetic Energy'
)

plt.plot(
    time_years,
    potential_energy_history,
    label='Potential Energy'
)

plt.plot(
    time_years,
    total_energy_history,
    label='Total Energy'
)

plt.xlabel('Time (years)')

plt.ylabel('Energy (J)')

plt.title(
    'Energy Conservation in N-Body Solar System Simulation'
)

plt.legend()

plt.grid(True)

plt.show()
plt.savefig("energy_conservation.pdf")

## ============================================================
## Relative Energy Error Plot
## ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    time_years,
    relative_energy_error
)

plt.xlabel('Time (years)')

plt.ylabel('Relative Energy Error')

plt.title(
    'Relative Energy Error'
)

plt.grid(True)

plt.show()
plt.savefig("relative_energy_error.pdf")

## ============================================================
## Centre of Mass Position Plot
## ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    time_years,
    com_position_history[:, 0] / AU,
    label='COM x'
)

plt.plot(
    time_years,
    com_position_history[:, 1] / AU,
    label='COM y'
)

plt.plot(
    time_years,
    com_position_history[:, 2] / AU,
    label='COM z'
)

plt.xlabel('Time (years)')

plt.ylabel('COM Position (AU)')

plt.title(
    'Centre of Mass Position vs Time'
)

plt.legend()

plt.grid(True)

plt.show()
plt.savefig("COM_position.pdf")







