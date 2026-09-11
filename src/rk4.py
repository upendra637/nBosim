import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
G=6.67430e-11  # gravitational constant in m^3 kg^-1 s^-2
M_sun = 1.989e30             # kg
AU = 1.495978707e11          # m

##computing acceleration
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

##compting the eqquations of motion
def derivatives(positions,velocities,masses):
    dr_dt = velocities
    dv_dt = calculate_acceleration(masses, positions)
    return dr_dt, dv_dt

###implementing the RK4 method to compute the velocities and positions of the bodies
def rk4_step(positions, velocities, masses, dt):
    #k1
    k1_r, k1_v = derivatives(positions, velocities, masses)
    #k2
    k2_r, k2_v = derivatives(positions + 0.5 * dt * k1_r, velocities + 0.5 * dt * k1_v, masses)
    #k3
    k3_r, k3_v = derivatives(positions + 0.5 * dt * k2_r, velocities + 0.5 * dt * k2_v, masses)
    #k4
    k4_r, k4_v = derivatives(positions + dt * k3_r, velocities + dt * k3_v, masses)

    new_positions = positions + (dt / 6) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
    new_velocities = velocities + (dt / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
    return new_positions, new_velocities

## Solar System Data
names=["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]
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

#converting masses to kg
masses = masses_solar * M_sun


#semi-major axes in Au
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

#converting semi-major axes to meters
semi_major_axes = semi_major_axes_AU * AU

#Eccentricities of the planets

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

#Inclinations of the planets in degrees
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



## inintial positions and velocities
n=len(masses)
positions = np.zeros((n, 3))
velocities = np.zeros((n, 3))
positions[0]=[0, 0, 0]  # Sun at origin
velocities[0]=[0, 0, 0]  # Sun stationary

mu=G * masses[0]  # gravitational parameter for the Sun

for i in range(1, n):
    r_perihelion = semi_major_axes[i] * (1 - eccentricities[i])
    v_perihelion = np.sqrt(mu * (2/r_perihelion - 1/semi_major_axes[i]))
    inc=inclinations[i]
    positions[i] = [r_perihelion, 0, 0] 
    velocities[i] = [0, v_perihelion*np.cos(inc), v_perihelion*np.sin(inc)]

##Orbital Periods
# Number of numerical steps per orbital period
steps_per_orbit = 1000


# Orbital periods
orbital_periods = np.zeros(n)


for i in range(1, n):

    # Kepler's third law
    #
    # T = 2*pi*sqrt(a^3 / GM)
    #

    orbital_periods[i] = (2 * np.pi* np.sqrt(semi_major_axes[i]**3/ (G * masses[0])))

#Planet specific time steps
dt_planets = np.zeros(n)
for i in range(1, n):
    dt_planets[i] = orbital_periods[i] / steps_per_orbit

#Display orbital periods and time steps
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

#Global time step for the simulation
dt = np.min(dt_planets[1:])  
print("\n--------------------------------------------")

print(
    f"Global timestep = "
    f"{dt / DAY if 'DAY' in globals() else dt/(24*3600):.4f} days"
)



## Simulation parameters
DAY=24*60*60
years=165
total_time=years*365.25*DAY  # total simulation time in seconds
steps=int(total_time/dt)
print(f"\nTotal simulation time = {years} years")

print(f"Total integration steps = {steps}")


trajectories = np.zeros((steps, n, 3))

for step in range(steps):
    trajectories[step] = positions
    new_positions, new_velocities = rk4_step(positions, velocities, masses, dt)
    positions, velocities = new_positions, new_velocities

#Converting trajectories to AU for plotting
trajectories_AU = trajectories / AU

##3D Plot the orbits of the planets
fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111,projection='3d')


# Plot planetary trajectories
for i in range(1, n):

    ax.plot(
        trajectories_AU[:, i, 0],
        trajectories_AU[:, i, 1],
        trajectories_AU[:, i, 2],
        label=names[i]
    )


# Plot Sun
ax.scatter(
    trajectories_AU[:, 0, 0],
    trajectories_AU[:, 0, 1],
    trajectories_AU[:, 0, 2],
    color='yellow',
    s=5,
    label='Sun'
)


# ============================================================
# Labels
# ============================================================

ax.set_title(
    '3D N-Body Solar System Simulation using RK4',
    fontsize=14
)

ax.set_xlabel('X position (AU)')

ax.set_ylabel('Y position (AU)')

ax.set_zlabel('Z position (AU)')


ax.legend()

plt.show()




