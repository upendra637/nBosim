import numpy as np
import matplotlib.pyplot as plt
G=6.67430e-11  # gravitational constant in m^3 kg^-1 s^-2

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
names=["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
masses=np.array([
    1.989e30,      # Sun
    3.301e23,      # Mercury
    4.867e24,      # Venus
    5.972e24,      # Earth
    6.417e23,      # Mars
    1.898e27,      # Jupiter
    5.683e26,      # Saturn
    8.681e25,      # Uranus
    1.024e26       # Neptune
])
semi_major_axes=np.array([
    0.0,
    5.791e10,
    1.082e11,
    1.496e11,
    2.279e11,
    7.785e11,
    1.434e12,
    2.871e12,
    4.495e12
])

eccentricities = np.array([
    0.0,
    0.2056,
    0.0068,
    0.0167,
    0.0934,
    0.0489,
    0.0565,
    0.0463,
    0.0095
])

## inintial positions and velocities
n=len(masses)
positions = np.zeros((n, 3))
velocities = np.zeros((n, 3))
positions[0]=[0, 0, 0]  # Sun at origin
velocities[0]=[0, 0, 0]  # Sun stationary

mu=G * masses[0]  # gravitational parameter for the Sun

for i in range(1, n):
    r_perihelion = semi_major_axes[i] * (1 - eccentricities[i])
    positions[i] = [r_perihelion, 0, 0]
    v_perihelion = np.sqrt(mu * (2/r_perihelion - 1/semi_major_axes[i]))
    velocities[i] = [0, v_perihelion, 0]


## Simulation parameters
DAY=24*60*60
dt=0.5*DAY  # time step in seconds
years=84
total_time=years*365.25*DAY  # total simulation time in seconds
steps=int(total_time/dt)

trajectories = np.zeros((steps, n, 3))

for step in range(steps):
    trajectories[step] = positions
    new_positions, new_velocities = rk4_step(positions, velocities, masses, dt)
    positions, velocities = new_positions, new_velocities

##Plotting the orbits of the planets
plt.figure(figsize=(10, 10))
for i in range(1, n):
    plt.plot(trajectories[:, i, 0], trajectories[:, i, 1], label=names[i])
plt.scatter(trajectories[:, 0, 0], trajectories[:, 0, 1], color='yellow', label='Sun', s=100)
plt.title('N_body Solar System Simulation using RK4 Method')
plt.xlabel('x position (m)')
plt.ylabel('y position (m)')
plt.legend()
plt.axis('equal')
plt.grid()
plt.show()  




