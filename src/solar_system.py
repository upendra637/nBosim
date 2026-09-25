"""
Animated 2D Solar System — real N-body gravity via RK4 integration,
Mercury through Pluto, with the Moon orbiting Earth in the same view.

How this differs from a Kepler-ellipse simulation:
  - Before, each planet's path was calculated with a formula that
    already assumes a perfect, undisturbed ellipse.
  - Here, every body (Sun + 9 planets) genuinely pulls on every other
    body via Newton's law of gravitation, and we step the whole
    system forward in time using 4th-order Runge-Kutta (RK4)
    integration. The elliptical orbits emerge from the physics
    instead of being assumed.
  - One side effect of real N-body gravity: the Sun itself wobbles
    very slightly, tugged around by Jupiter and friends. That's not a
    bug, that's real physics (it's how we actually detect exoplanets
    around other stars).
  - The Moon is still handled as a simple circular overlay around
    Earth, not as part of the N-body gravity solve. Its real orbit
    (27 days) is much faster than the step size used for a smooth
    60-year planetary simulation, so including it properly would need
    a much finer time step just for the Earth-Moon pair. Its radius
    is also exaggerated for visibility, same as before.

Run this in Jupyter for the inline animation, or use the save block
at the bottom to export an mp4/gif.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# ------------------------------------------------------------------
# 1. Bodies: Sun + Mercury through Pluto
# ------------------------------------------------------------------
# Units used throughout the physics: AU for distance, years for time,
# solar masses for mass. In these units, G = 4*pi^2 (this falls out
# of Kepler's third law for Earth's orbit: a=1 AU, period=1 yr).

names = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter",
         "Saturn", "Uranus", "Neptune", "Pluto"]

# mass, in solar masses (Sun = 1.0)
mass = np.array([
    1.0,
    1.6601e-7,   # Mercury
    2.4478e-6,   # Venus
    3.0035e-6,   # Earth
    3.2131e-7,   # Mars
    9.5479e-4,   # Jupiter
    2.8589e-4,   # Saturn
    4.3662e-5,   # Uranus
    5.1514e-5,   # Neptune
    6.5500e-9,   # Pluto
])

# orbital elements, used only to build STARTING conditions --
# after that, gravity alone decides where everything goes
a = np.array([0.0, 0.3871, 0.7233, 1.0000, 1.5237,
              5.2028, 9.5388, 19.1914, 30.0611, 39.4817])       # AU
e = np.array([0.0, 0.2056, 0.0068, 0.0167, 0.0934,
              0.0489, 0.0565, 0.0463, 0.0086, 0.2488])          # eccentricity
phase = np.array([0.0, 0.0, 0.7, 1.2, 2.0,
                   2.7, 3.5, 4.0, 5.0, 1.5])                    # starting angle, radians

colors = ["gold", "#b1b1b1", "#e8c39e", "#3f83ff", "#c1440e",
          "#d9a066", "#e3c16f", "#8fd1e0", "#4166f5", "#c9a0dc"]

n_bodies = len(names)
earth_idx = names.index("Earth")
sun_idx = names.index("Sun")

G = 4 * np.pi**2   # gravitational constant in AU^3 / (yr^2 * solar mass)

# ------------------------------------------------------------------
# 2. Build starting positions and velocities
# ------------------------------------------------------------------
# Each planet starts at its closest approach to the Sun (perihelion),
# at distance a*(1-e), moving perpendicular to that radius. The speed
# there comes from the vis-viva equation, which tells you how fast a
# body must move at a given distance to trace out a given ellipse.

pos0 = np.zeros((n_bodies, 2))
vel0 = np.zeros((n_bodies, 2))

for i in range(1, n_bodies):   # skip the Sun, index 0, which starts at rest
    r0 = a[i] * (1 - e[i])
    direction = np.array([np.cos(phase[i]), np.sin(phase[i])])
    tangent = np.array([-np.sin(phase[i]), np.cos(phase[i])])

    pos0[i] = r0 * direction

    # vis-viva equation: speed needed at distance r0 on an ellipse of
    # size a, orbiting a mass GM (here just the Sun's, since it vastly
    # outweighs everything else)
    GM_sun = G * mass[sun_idx]
    speed = np.sqrt(GM_sun * (2 / r0 - 1 / a[i]))
    vel0[i] = speed * tangent

# Give the Sun a small initial velocity that exactly cancels the total
# momentum of the planets. Without this, the whole system's center of
# mass quietly drifts sideways forever (every planet nudges the Sun a
# little via gravity, and those nudges don't average out to zero if
# the Sun starts truly at rest). With this correction, the Sun still
# wobbles slightly as real N-body physics demands -- it just wobbles
# in place instead of sliding off across the picture.
total_planet_momentum = np.sum(mass[1:, None] * vel0[1:], axis=0)
vel0[sun_idx] = -total_planet_momentum / mass[sun_idx]

# ------------------------------------------------------------------
# 3. Gravity: acceleration felt by every body from every other body
# ------------------------------------------------------------------
softening = 1e-6   # tiny buffer so nothing divides by zero if two bodies coincide

def acceleration(pos):
    """Given every body's position (n_bodies x 2), return every body's
    gravitational acceleration (n_bodies x 2), from Newton's law of
    gravitation applied pairwise between all bodies at once."""
    n = pos.shape[0]
    acc = np.zeros_like(pos)

    for i in range(n):
        diff = pos - pos[i]                          # vectors from body i to all others
        dist_sq = np.sum(diff**2, axis=1) + softening
        dist_cubed = dist_sq ** 1.5
        dist_cubed[i] = np.inf                        # a body doesn't pull on itself

        acc[i] = G * np.sum((mass[:, None] * diff) / dist_cubed[:, None], axis=0)

    return acc

# ------------------------------------------------------------------
# 4. RK4 integration -- step the whole system forward in time
# ------------------------------------------------------------------
def rk4_step(pos, vel, dt):
    """Advance every body's position and velocity by one time step dt,
    using 4th-order Runge-Kutta. Gravity only depends on position, so
    the four stages sample the acceleration at four slightly different
    predicted positions within the step, then blend them for accuracy."""

    k1v = vel
    k1a = acceleration(pos)

    k2v = vel + 0.5 * dt * k1a
    k2a = acceleration(pos + 0.5 * dt * k1v)

    k3v = vel + 0.5 * dt * k2a
    k3a = acceleration(pos + 0.5 * dt * k2v)

    k4v = vel + dt * k3a
    k4a = acceleration(pos + dt * k3v)

    pos_new = pos + (dt / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)
    vel_new = vel + (dt / 6) * (k1a + 2 * k2a + 2 * k3a + k4a)

    return pos_new, vel_new

# ------------------------------------------------------------------
# 5. Run the simulation: many small RK4 steps, keep every Nth one
# ------------------------------------------------------------------
simulation_years = 60
frames = 1500
substeps_per_frame = 20          # fine integration steps between each displayed frame
total_steps = frames * substeps_per_frame
dt = simulation_years / total_steps   # roughly 0.7 days per integration step

pos = pos0.copy()
vel = vel0.copy()

trajectory = np.zeros((frames, n_bodies, 2))   # only the frames we'll actually display

frame_i = 0
for step in range(total_steps):
    pos, vel = rk4_step(pos, vel, dt)
    if (step + 1) % substeps_per_frame == 0:
        trajectory[frame_i] = pos
        frame_i += 1

# reshape for convenience: body_x[i] is body i's x across all frames
body_x = trajectory[:, :, 0].T
body_y = trajectory[:, :, 1].T

earth_x, earth_y = body_x[earth_idx], body_y[earth_idx]

# ------------------------------------------------------------------
# 6. Moon position -- simple circular overlay around Earth (not N-body)
# ------------------------------------------------------------------
# We need the Moon's angle around Earth at every frame -- that part is
# shared by both views below. What differs is the radius:
#   - moon_a_real is the Moon's true distance, 0.00257 AU. At the
#     scale of the whole solar system (tens of AU across) that's a
#     fraction of a pixel, so it's only useful zoomed way in.
#   - moon_a_display is a deliberately exaggerated radius so a small
#     ring is still visible next to Earth in the full solar-system
#     view, purely as a "something orbits here" indicator.
moon_period_years = 27.321661 / 365.25
moon_a_real = 0.00257             # AU -- the Moon's true orbital radius
moon_a_display = 0.35             # AU -- exaggerated, main-view only

time_years = np.linspace(0, simulation_years, frames)
moon_angle = 2 * np.pi * time_years / moon_period_years

# true-scale position, relative to Earth (used by the zoomed inset)
moon_relative_real_x = moon_a_real * np.cos(moon_angle)
moon_relative_real_y = moon_a_real * np.sin(moon_angle)

# exaggerated position, relative to Earth (used by the main view)
moon_relative_display_x = moon_a_display * np.cos(moon_angle)
moon_relative_display_y = moon_a_display * np.sin(moon_angle)

moon_x = earth_x + moon_relative_display_x
moon_y = earth_y + moon_relative_display_y

# ------------------------------------------------------------------
# 7. Figure setup
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 11))
fig.patch.set_facecolor("black")
ax.set_facecolor("black")

ax.set_xlim(-42, 42)
ax.set_ylim(-42, 42)
ax.set_aspect("equal")
ax.set_xlabel("X (AU)", color="white")
ax.set_ylabel("Y (AU)", color="white")
ax.set_title("Solar System (RK4 N-body) with the Moon", color="white")
ax.tick_params(colors="white")
ax.grid(True, alpha=0.15, color="gray")

# faint trail showing where each body has actually been, from the simulation
for i in range(n_bodies):
    ax.plot(body_x[i], body_y[i], linestyle="--", alpha=0.25, color=colors[i])

# moving dot per body (Sun included -- it wobbles slightly, that's real)
body_points = []
for i in range(n_bodies):
    if i == earth_idx:
        p, = ax.plot([], [], "o", markersize=12, color=colors[i],
                      markeredgecolor="white", markeredgewidth=1.5,
                      label=names[i], zorder=5)
    elif i == sun_idx:
        p, = ax.plot([], [], "o", markersize=16, color=colors[i], label=names[i])
    else:
        p, = ax.plot([], [], "o", markersize=6, color=colors[i], label=names[i])
    body_points.append(p)

earth_label = ax.text(0, 0, "EARTH", color="white", fontsize=9,
                       fontweight="bold", ha="left", va="bottom")

moon_point, = ax.plot([], [], "o", markersize=5, color="lightgray",
                       markeredgecolor="white", label="Moon", zorder=5)
moon_orbit_line, = ax.plot([], [], linestyle=":", alpha=0.6, color="lightgray")

ax.legend(loc="upper right", fontsize=8, ncol=2, facecolor="black",
          edgecolor="gray", labelcolor="white")

# ------------------------------------------------------------------
# 8. Animation update function
# ------------------------------------------------------------------
theta_ring = np.linspace(0, 2 * np.pi, 150)

def update(frame):
    for i in range(n_bodies):
        body_points[i].set_data([body_x[i, frame]], [body_y[i, frame]])

    ex, ey = earth_x[frame], earth_y[frame]
    earth_label.set_position((ex + 1.0, ey + 1.0))

    moon_point.set_data([moon_x[frame]], [moon_y[frame]])

    orbit_x = ex + moon_a_display * np.cos(theta_ring)
    orbit_y = ey + moon_a_display * np.sin(theta_ring)
    moon_orbit_line.set_data(orbit_x, orbit_y)

    return body_points + [moon_point, moon_orbit_line, earth_label]


# ------------------------------------------------------------------
# 9. Build and show the animation
# ------------------------------------------------------------------
animation = FuncAnimation(fig, update, frames=frames, interval=15, blit=True)
plt.close()

HTML(animation.to_jshtml())

# ------------------------------------------------------------------
# Optional: save to a file instead of / in addition to viewing inline
# ------------------------------------------------------------------
# animation.save("solar_system.mp4", writer="ffmpeg", fps=45)
# animation.save("solar_system.gif", writer="pillow", fps=45)