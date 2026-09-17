import numpy as np

# Acceleration
def acceleration(masses, x_pos, y_pos, z_pos, G=39.478):
    """
    Calculate the acceleration of each mass due to gravitational forces
    from all other masses.

    Parameters:
    masses (list): List of masses.
    x_pos (list): List of x positions of the masses.
    y_pos (list): List of y positions of the masses.
    z_pos (list): List of z positions of the masses.
    G (float): Gravitational constant.

    Returns:
    ax (list): List of x accelerations for each mass.
    ay (list): List of y accelerations for each mass.
    az (list): List of z accelerations for each mass.
    """

    n = len(masses)

    ax = np.zeros(n)
    ay = np.zeros(n)
    az = np.zeros(n)


    for i in range(n):

        Fx, Fy, Fz = 0, 0, 0

        for j in range(n):

            if i != j:

                dx = x_pos[j] - x_pos[i]
                dy = y_pos[j] - y_pos[i]
                dz = z_pos[j] - z_pos[i]

                r = np.sqrt(dx**2 + dy**2 + dz**2)

                Fx += G * masses[i] * masses[j] * dx / (r**3)
                Fy += G * masses[i] * masses[j] * dy / (r**3)
                Fz += G * masses[i] * masses[j] * dz / (r**3)

        ax[i] += Fx / masses[i]
        ay[i] += Fy / masses[i]
        az[i] += Fz / masses[i]

    return ax, ay, az


# velocity
def velocity(masses, x_pos, y_pos, z_pos, ux, uy, uz, dt, G=39.478):
    """
    Calculate the velocity of each mass based on its acceleration.

    Returns:
    vx (list): List of x velocities for each mass.
    vy (list): List of y velocities for each mass.
    vz (list): List of z velocities for each mass.
    """

    ax, ay, az = acceleration(
        masses,
        x_pos,
        y_pos,
        z_pos,
        G
    )


    vx = ux + ax * dt
    vy = uy + ay * dt
    vz = uz + az * dt


    return vx, vy, vz


# position
def position(masses, x_pos, y_pos, z_pos, ux, uy, uz, dt, G=39.478):
    """
    Calculate the position of each mass based on its velocity.

    Returns:
    x_pos (list): Updated x positions.
    y_pos (list): Updated y positions.
    z_pos (list): Updated z positions.
    ux (list): Updated x velocities.
    uy (list): Updated y velocities.
    uz (list): Updated z velocities.
    """

    vx, vy, vz = velocity(
        masses,
        x_pos,
        y_pos,
        z_pos,
        ux,
        uy,
        uz,
        dt,
        G
    )

    x = x_pos + vx * dt
    y = y_pos + vy * dt
    z = z_pos + vz * dt


    # Update the current position and velocity
    x_pos = x
    y_pos = y
    z_pos = z

    ux = vx
    uy = vy
    uz = vz


    return x_pos, y_pos, z_pos, ux, uy, uz


# Center of mass
def com_frame(masses, x_pos, y_pos, z_pos, ux, uy, uz):

    """
    Calculate the center of mass position and velocity
    for the system of masses.
    """

    total_mass = np.sum(masses)

    com_x = np.sum(masses * x_pos) / total_mass
    com_y = np.sum(masses * y_pos) / total_mass
    com_z = np.sum(masses * z_pos) / total_mass

    com_vx = np.sum(masses * ux) / total_mass
    com_vy = np.sum(masses * uy) / total_mass
    com_vz = np.sum(masses * uz) / total_mass

    return com_x, com_y, com_z, com_vx, com_vy, com_vz


# Position relative to center of mass
def relative_position(masses, x_pos, y_pos, z_pos, ux, uy, uz):

    """
    Calculate the position and velocity of each mass
    relative to the center of mass.
    """

    com_x, com_y, com_z, com_vx, com_vy, com_vz = com_frame(
        masses,
        x_pos,
        y_pos,
        z_pos,
        ux,
        uy,
        uz
    )

    x_relative = x_pos - com_x
    y_relative = y_pos - com_y
    z_relative = z_pos - com_z

    vx_relative = ux - com_vx
    vy_relative = uy - com_vy
    vz_relative = uz - com_vz

    return (
        x_relative,
        y_relative,
        z_relative,
        vx_relative,
        vy_relative,
        vz_relative
    )


# Energy
def energy(masses, x_pos, y_pos, z_pos, ux, uy, uz, G=39.478):

    """
    Calculate the total kinetic energy, gravitational potential energy
    and total energy of the system.
    """

    n = len(masses)

    # Kinetic energy
    kinetic_energy = 0

    for i in range(n):

        velocity_squared = (
            ux[i]**2 +
            uy[i]**2 +
            uz[i]**2
        )

        kinetic_energy += 0.5 * masses[i] * velocity_squared


    # Potential energy
    potential_energy = 0

    for i in range(n):

        for j in range(i + 1, n):

            dx = x_pos[j] - x_pos[i]
            dy = y_pos[j] - y_pos[i]
            dz = z_pos[j] - z_pos[i]

            r = np.sqrt(
                dx**2 +
                dy**2 +
                dz**2
            )

            potential_energy -= (
                G * masses[i] * masses[j] / r
            )


    # Total energy
    total_energy = kinetic_energy + potential_energy

    return kinetic_energy, potential_energy, total_energy









# ==========================================================================================
import os
import pandas as pd

# masses (solar masses)
masses = np.array(
    [1.0,
    1.660e-7,
    2.448e-6,
    3.003e-6,
    3.227e-7,
    9.545e-4,
    2.857e-4,
    4.366e-5,
    5.151e-5,
    6.55e-9]
)


# average orbital distances from Sun (AU)
X = np.array(
    [0.0,
     0.387,
     0.723,
     1.000,
     1.524,
     5.203,
     9.537,
     19.191,
     30.07,
     39.48]
)

y = np.zeros_like(X)
z = np.zeros_like(X)


# average orbital velocities (AU/year)
uy = np.array(
    [0.0,
     10.07,
     7.39,
     6.28,
     5.09,
     2.62,
     2.12,
     1.48,
     1.14,
     0.90]
)

ux = np.zeros_like(X)
uz = np.zeros_like(X)


# orbital inclinations (degrees)
inclination = np.array(
    [0.0,
     7.00,
     3.39,
     0.00,
     1.85,
     1.31,
     2.49,
     0.77,
     1.77,
     17.15]
)


# Convert inclination from degrees to radians
inclination = np.radians(inclination)


# Resolve the initial velocity into y and z components
uz = uy * np.sin(inclination)
uy = uy * np.cos(inclination)

dt = 0.001
N = 200000


# Store positions
x_pos = np.zeros((len(masses), N))
y_pos = np.zeros((len(masses), N))
z_pos = np.zeros((len(masses), N))

# Store velocities
ux_data = np.zeros((len(masses), N))
uy_data = np.zeros((len(masses), N))
uz_data = np.zeros((len(masses), N))


for i in range(N):

    X, Y, Z, ux, uy, uz = position(
        masses,
        X,
        y,
        z,
        ux,
        uy,
        uz,
        dt
    )

    x_pos[:, i] = X
    y_pos[:, i] = Y
    z_pos[:, i] = Z

    ux_data[:, i] = ux
    uy_data[:, i] = uy
    uz_data[:, i] = uz

    y = Y
    z = Z


# Calculate positions relative to center of mass
X_xom = np.zeros_like(x_pos)
Y_xom = np.zeros_like(y_pos)
Z_xom = np.zeros_like(z_pos)

# velocity relative to center of mass
vx = np.zeros_like(x_pos)
vy = np.zeros_like(y_pos)
vz = np.zeros_like(z_pos)

# calculate positions relative to center of mass
for i in range(N):

    X_xom[:, i], Y_xom[:, i], Z_xom[:, i], vx[:, i], vy[:, i], vz[:, i] = relative_position(
        masses,
        x_pos[:, i],
        y_pos[:, i],
        z_pos[:, i],
        ux_data[:, i],
        uy_data[:, i],
        uz_data[:, i]
    )

# Calculate energy
kinetic_energy = np.zeros(N)
potential_energy = np.zeros(N)
total_energy = np.zeros(N)

for i in range(N):

    kinetic_energy[i], potential_energy[i], total_energy[i] = energy(
        masses,
        x_pos[:, i],
        y_pos[:, i],
        z_pos[:, i],
        vx[:, i],
        vy[:, i],
        vz[:, i]
    )


labels = [
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


# saving the data to a csv file

os.makedirs("Data", exist_ok=True)

data = {}

for i in range(len(masses)):
    data[labels[i] + "_x"] = x_pos[i, :]
    data[labels[i] + "_y"] = y_pos[i, :]
    data[labels[i] + "_z"] = z_pos[i, :]

df = pd.DataFrame(data)

df.to_csv("Data/positions.csv", index=False)

# Save energy data to CSV

energy_data = pd.DataFrame({
    "Kinetic Energy": kinetic_energy,
    "Potential Energy": potential_energy,
    "Total Energy": total_energy
})

energy_data.to_csv("Data/energy.csv", index=False)


