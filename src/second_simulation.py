import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import rebound


# In REBOUND's Solar System units:
# distance = AU, mass = Solar masses, time = years / (2π)
# Therefore, G = 1.
G = 1.0


def create_initial_state():
    """Load planets and add Jupiter's four Galilean moons."""
    sim = rebound.Simulation()
    sim.add("solar system")

    jupiter = sim.particles[5]

    # Pluto is not in REBOUND's built-in Solar System dataset.
    sim.add(
        m=6.55e-9,
        a=39.482,
        e=0.2488,
        inc=np.radians(17.16),
        Omega=np.radians(110.30),
        omega=np.radians(113.76),
        primary=sim.particles[0]
    )

    # Jupiter's four largest moons
    sim.add(m=4.49e-8, a=0.002818, e=0.0041, primary=jupiter)  # Io
    sim.add(m=2.41e-8, a=0.004486, e=0.0094, primary=jupiter)  # Europa
    sim.add(m=7.45e-8, a=0.007155, e=0.0013, primary=jupiter)  # Ganymede
    sim.add(m=5.41e-8, a=0.012585, e=0.0074, primary=jupiter)  # Callisto

    sim.move_to_com()

    masses = np.array([particle.m for particle in sim.particles])

    positions = np.array([
        [particle.x, particle.y, particle.z]
        for particle in sim.particles
    ])

    velocities = np.array([
        [particle.vx, particle.vy, particle.vz]
        for particle in sim.particles
    ])

    return masses, positions, velocities


def calculate_acceleration(positions, masses):
    """Calculate Newtonian gravitational acceleration on every body."""
    # displacement[i, j] = position of j - position of i
    displacement = positions[None, :, :] - positions[:, None, :]

    distance_squared = np.sum(displacement ** 2, axis=2)

    # Avoid division by zero for self-interaction.
    inverse_distance_cubed = np.zeros_like(distance_squared)
    nonzero = distance_squared > 0

    inverse_distance_cubed[nonzero] = (
        1.0 / distance_squared[nonzero] ** 1.5
    )

    acceleration = G * np.sum(
        displacement
        * inverse_distance_cubed[:, :, None]
        * masses[None, :, None],
        axis=1
    )

    return acceleration


def rk4_step(positions, velocities, masses, dt):
    """Advance all bodies one step using fourth-order Runge-Kutta."""

    # k1
    k1_position = velocities
    k1_velocity = calculate_acceleration(positions, masses)

    # k2
    k2_position = velocities + 0.5 * dt * k1_velocity
    k2_velocity = calculate_acceleration(
        positions + 0.5 * dt * k1_position,
        masses
    )

    # k3
    k3_position = velocities + 0.5 * dt * k2_velocity
    k3_velocity = calculate_acceleration(
        positions + 0.5 * dt * k2_position,
        masses
    )

    # k4
    k4_position = velocities + dt * k3_velocity
    k4_velocity = calculate_acceleration(
        positions + dt * k3_position,
        masses
    )

    new_positions = positions + (
        dt / 6.0
    ) * (
        k1_position
        + 2 * k2_position
        + 2 * k3_position
        + k4_position
    )

    new_velocities = velocities + (
        dt / 6.0
    ) * (
        k1_velocity
        + 2 * k2_velocity
        + 2 * k3_velocity
        + k4_velocity
    )

    return new_positions, new_velocities


def total_energy(positions, velocities, masses):
    """Calculate total kinetic plus gravitational potential energy."""
    kinetic_energy = 0.5 * np.sum(
        masses * np.sum(velocities ** 2, axis=1)
    )

    potential_energy = 0.0

    for i in range(len(masses)):
        for j in range(i + 1, len(masses)):
            distance = np.linalg.norm(positions[j] - positions[i])
            potential_energy -= G * masses[i] * masses[j] / distance

    return kinetic_energy + potential_energy


def integrate_rk4(positions, velocities, masses, target_time, current_time, dt):
    """Use RK4 steps until the requested target time is reached."""
    while current_time < target_time:
        step = min(dt, target_time - current_time)

        positions, velocities = rk4_step(
            positions,
            velocities,
            masses,
            step
        )

        current_time += step

    return positions, velocities, current_time


def visual_scale(x, y):
    """Compress outer distances for clearer display."""
    radius = np.hypot(x, y)

    scale = np.divide(
        np.sqrt(radius),
        radius,
        out=np.zeros_like(radius, dtype=float),
        where=radius != 0
    )

    return x * scale, y * scale


# -------------------------------------------------
# RK4 Solar System simulation: 165 years
# -------------------------------------------------

years = 165
frames = 1000
tmax = years * 2 * math.pi

# Small RK4 step: about 2.8 hours
dt = 0.0005

masses, positions, velocities = create_initial_state()
initial_energy = total_energy(positions, velocities, masses)

solar_times = np.linspace(0, tmax, frames)
solar_positions = np.zeros((frames, 10, 2))

current_time = 0.0

for frame, target_time in enumerate(solar_times):
    positions, velocities, current_time = integrate_rk4(
        positions,
        velocities,
        masses,
        target_time,
        current_time,
        dt
    )

    # Save Sun, eight planets, and Pluto for display
    solar_positions[frame] = positions[:10, :2]

final_energy = total_energy(positions, velocities, masses)
relative_energy_error = abs(
    (final_energy - initial_energy) / initial_energy
)


# -----------------------
# --------------------------
# RK4 Moon simulation: 30 days relative to Jupiter
# -------------------------------------------------

moon_days = 30
moon_tmax = moon_days * 2 * math.pi / 365.25

moon_masses, moon_positions, moon_velocities = create_initial_state()

moon_times = np.linspace(0, moon_tmax, frames)
moon_relative_positions = np.zeros((frames, 4, 2))

moon_current_time = 0.0

for frame, target_time in enumerate(moon_times):
    moon_positions, moon_velocities, moon_current_time = integrate_rk4(
        moon_positions,
        moon_velocities,
        moon_masses,
        target_time,
        moon_current_time,
        dt
    )

    # Jupiter = 5; Pluto = 9; moons = 10, 11, 12, 13
    for moon_number in range(4):
        moon_relative_positions[frame, moon_number] = (
            moon_positions[10 + moon_number, :2]
            - moon_positions[5, :2]
        )


with open("energy.txt", "w") as file:
    file.write("Numerical method: RK4\n")
    file.write(f"Simulation length: {years} years\n")
    file.write(f"Relative energy error: {relative_energy_error:e}\n")


# -------------------------------------------------
# One animation plot
# -------------------------------------------------

moon_display_scale = 250

fig, ax = plt.subplots(figsize=(10, 10), facecolor="black")
ax.set_facecolor("#050814")

ax.set_title("Solar System — RK4 Method")
ax.set_xlabel("Visual x position")
ax.set_ylabel("Visual y position")
ax.set_xlim(-7.5, 7.5)
ax.set_ylim(-7.5, 7.5)
ax.set_aspect("equal")
ax.grid(color="#64748b", alpha=0.35)
ax.tick_params(colors="white")

for spine in ax.spines.values():
    spine.set_color("white")

ax.title.set_color("white")
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")

planet_names = [
    "Sun", "Mercury", "Venus", "Earth", "Mars",
    "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"
]

planet_colors = [
    "gold", "gray", "orange", "blue", "red",
    "peru", "goldenrod", "cyan", "deepskyblue", "magenta"
]

# Visual marker sizes (not to physical scale).
planet_sizes = [
    20,  # Sun
    2,   # Mercury
    4,   # Venus
    4,   # Earth
    3,   # Mars
    8,   # Jupiter
    7,   # Saturn
    5,   # Uranus
    5,   # Neptune
    2    # Pluto
]

moon_names = ["Io", "Europa", "Ganymede", "Callisto"]
moon_colors = ["white", "lightgray", "khaki", "silver"]

planet_dots = []
planet_trails = []

for i in range(10):
    trail, = ax.plot(
        [],
        [],
        color=planet_colors[i],
        linewidth=0.8,
        alpha=0.65
    )

    dot, = ax.plot(
        [],
        [],
        "o",
        color=planet_colors[i],
        markersize=planet_sizes[i],
        label=planet_names[i]
    )

    planet_dots.append(dot)
    planet_trails.append(trail)

moon_dots = []
moon_trails = []

for i in range(4):
    trail, = ax.plot(
        [],
        [],
        color=moon_colors[i],
        linewidth=1,
        alpha=0.8
    )

    dot, = ax.plot(
        [],
        [],
        "o",
        color=moon_colors[i],
        markersize=2,
        label=moon_names[i]
    )

    moon_dots.append(dot)
    moon_trails.append(trail)

legend = ax.legend(loc="upper right", fontsize="small")
legend.get_frame().set_facecolor("#111827")
legend.get_frame().set_edgecolor("white")

for text in legend.get_texts():
    text.set_color("white")

time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, color="white")

note_text = ax.text(
    0.02,
    0.02,
    "RK4 integration; moon paths shown relative to Jupiter",
    transform=ax.transAxes,
    fontsize=8,
    color="white"
)


def update(frame):
    artists = []

    # Draw Sun and planets
    for i in range(10):
        x, y = visual_scale(
            solar_positions[frame, i, 0],
            solar_positions[frame, i, 1]
        )

        trail_x, trail_y = visual_scale(
            solar_positions[:frame + 1, i, 0],
            solar_positions[:frame + 1, i, 1]
        )

        planet_dots[i].set_data([x], [y])
        planet_trails[i].set_data(trail_x, trail_y)

        artists.append(planet_dots[i])
        artists.append(planet_trails[i])

    # Use Jupiter's current location for the moon coordinate system
    jupiter_x = solar_positions[frame, 5, 0]
    jupiter_y = solar_positions[frame, 5, 1]

    # Draw moon paths relative to Jupiter, not relative to the Sun
    for i in range(4):
        moon_x = (
            jupiter_x
            + moon_relative_positions[frame, i, 0] * moon_display_scale
        )

        moon_y = (
            jupiter_y
            + moon_relative_positions[frame, i, 1] * moon_display_scale
        )

        x, y = visual_scale(moon_x, moon_y)
        moon_dots[i].set_data([x], [y])

        trail_x, trail_y = visual_scale(
            jupiter_x
            + moon_relative_positions[:frame + 1, i, 0]
            * moon_display_scale,
            jupiter_y
            + moon_relative_positions[:frame + 1, i, 1]
            * moon_display_scale
        )

        moon_trails[i].set_data(trail_x, trail_y)

        artists.append(moon_dots[i])
        artists.append(moon_trails[i])

    solar_year = solar_times[frame] / (2 * math.pi)
    time_text.set_text(f"Solar System time: {solar_year:.1f} years")

    artists.append(time_text)
    artists.append(note_text)

    return artists


animation = FuncAnimation(
    fig,
    update,
    frames=frames,
    interval=25,
    blit=True
)

print("RK4 simulation complete.")
print(f"Relative energy error: {relative_energy_error:e}")
print("Results saved to energy.txt")

plt.show()
