import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ============================================================
# 1. CONSTANTS
# ============================================================

G = 6.67430e-11

M_earth = 5.972e24
M_moon = 7.342e22


# ============================================================
# 2. INITIAL CONDITIONS
# ============================================================

# Earth
x_earth = 0.0
y_earth = 0.0

vx_earth = 0.0
vy_earth = 0.0


# Moon
x_moon = 3.844e8
y_moon = 0.0

vx_moon = 0.0
vy_moon = 1.022e3


# ============================================================
# 3. TIME SETTINGS
# ============================================================

dt = 1000          # time step in seconds
steps = 30000      # number of steps

time = np.arange(steps) * dt


# ============================================================
# 4. ARRAYS TO STORE POSITIONS
# ============================================================

earth_x = np.zeros(steps)
earth_y = np.zeros(steps)

moon_x = np.zeros(steps)
moon_y = np.zeros(steps)


# Initial positions
earth_x[0] = x_earth
earth_y[0] = y_earth

moon_x[0] = x_moon
moon_y[0] = y_moon


# ============================================================
# 5. TWO-BODY SIMULATION
# ============================================================

for i in range(steps - 1):

    # --------------------------------------------------------
    # Distance between Earth and Moon
    # --------------------------------------------------------

    dx = moon_x[i] - earth_x[i]
    dy = moon_y[i] - earth_y[i]

    r = np.sqrt(dx**2 + dy**2)


    # --------------------------------------------------------
    # Gravitational force
    # --------------------------------------------------------

    F_x = G * M_earth * M_moon * dx / r**3
    F_y = G * M_earth * M_moon * dy / r**3


    # --------------------------------------------------------
    # Accelerations
    # --------------------------------------------------------

    # Earth
    ax_earth = F_x / M_earth
    ay_earth = F_y / M_earth

    # Moon
    ax_moon = -F_x / M_moon
    ay_moon = -F_y / M_moon


    # --------------------------------------------------------
    # Velocities
    # --------------------------------------------------------

    vx_earth += ax_earth * dt
    vy_earth += ay_earth * dt

    vx_moon += ax_moon * dt
    vy_moon += ay_moon * dt


    # --------------------------------------------------------
    # Positions
    # --------------------------------------------------------

    earth_x[i + 1] = earth_x[i] + vx_earth * dt
    earth_y[i + 1] = earth_y[i] + vy_earth * dt

    moon_x[i + 1] = moon_x[i] + vx_moon * dt
    moon_y[i + 1] = moon_y[i] + vy_moon * dt


# ============================================================
# 6. CREATE ANIMATION
# ============================================================

fig, ax = plt.subplots(figsize=(8, 8))


# Determine plot limits
max_range = 1.2 * np.max(
    np.sqrt(moon_x**2 + moon_y**2)
)


ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)

ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")

ax.set_title("Two-Body Problem: Earth–Moon")

ax.grid(True)
ax.set_aspect("equal")


# ============================================================
# 7. OBJECTS THAT WILL MOVE
# ============================================================

# Earth
earth_plot, = ax.plot(
    [],
    [],
    'o',
    markersize=12,
    label="Earth"
)


# Moon
moon_plot, = ax.plot(
    [],
    [],
    'o',
    markersize=6,
    label="Moon"
)


# Moon orbit trail
orbit_line, = ax.plot(
    [],
    [],
    '-',
    linewidth=1
)


# Time text
time_text = ax.text(
    0.02,
    0.95,
    '',
    transform=ax.transAxes
)


ax.legend()


# ============================================================
# 8. INITIALIZATION FUNCTION
# ============================================================

def init():

    earth_plot.set_data([], [])

    moon_plot.set_data([], [])

    orbit_line.set_data([], [])

    time_text.set_text('')

    return (
        earth_plot,
        moon_plot,
        orbit_line,
        time_text
    )


# ============================================================
# 9. ANIMATION UPDATE FUNCTION
# ============================================================

def update(frame):

    # Current position of Earth
    earth_plot.set_data(
        [earth_x[frame]],
        [earth_y[frame]]
    )


    # Current position of Moon
    moon_plot.set_data(
        [moon_x[frame]],
        [moon_y[frame]]
    )


    # Draw Moon's path
    orbit_line.set_data(
        moon_x[:frame + 1],
        moon_y[:frame + 1]
    )


    # Convert seconds → days
    days = time[frame] / (24 * 3600)


    time_text.set_text(
        f"Time = {days:.1f} days"
    )


    return (
        earth_plot,
        moon_plot,
        orbit_line,
        time_text
    )


# ============================================================
# 10. CREATE ANIMATION
# ============================================================

ani = FuncAnimation(
    fig,
    update,
    frames=steps,
    init_func=init,
    interval=20,
    blit=True
)


plt.show()