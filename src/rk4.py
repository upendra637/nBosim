import numpy as np
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



