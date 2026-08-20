import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# ============================================================
# 1. CONSTANTS
# ============================================================

G = 6.67430e-11

M_sun = 1.989e30
M_mercury = 3.301e23


# ============================================================
# 2. INITIAL CONDITIONS
# ============================================================

# Sun
x_sun = 0.0
y_sun = 0.0

vx_sun = 0.0
vy_sun = 0.0


# Mercury
x_mercury = 5.79e10
y_mercury = 0.0

vx_mercury = 0.0
vy_mercury = 4.78e4


# ============================================================
# 3. TIME SETTINGS
# ============================================================

dt = 10000          # time step in seconds
steps = 30000       # number of steps

time = np.arange(steps) * dt


# ============================================================
# 4. ARRAYS TO STORE POSITIONS
# ============================================================

sun_x = np.zeros(steps)
sun_y = np.zeros(steps)

mercury_x = np.zeros(steps)
mercury_y = np.zeros(steps)


# Initial positions
sun_x[0] = x_sun
sun_y[0] = y_sun

mercury_x[0] = x_mercury
mercury_y[0] = y_mercury


# ============================================================
# 5. TWO-BODY SIMULATION
# ============================================================

for i in range(steps - 1):

    # --------------------------------------------------------
    # Distance between Sun and Mercury
    # --------------------------------------------------------

    dx = mercury_x[i] - sun_x[i]
    dy = mercury_y[i] - sun_y[i]

    r = np.sqrt(dx**2 + dy**2)


    # --------------------------------------------------------
    # Gravitational force
    # --------------------------------------------------------

    F_x = G * M_sun * M_mercury * dx / r**3
    F_y = G * M_sun * M_mercury * dy / r**3


    # --------------------------------------------------------
    # Accelerations
    # --------------------------------------------------------

    # Sun
    ax_sun = F_x / M_sun
    ay_sun = F_y / M_sun

    # Mercury
    ax_mercury = -F_x / M_mercury
    ay_mercury = -F_y / M_mercury


    # --------------------------------------------------------
    # Velocities
    # --------------------------------------------------------

    vx_sun += ax_sun * dt
    vy_sun += ay_sun * dt

    vx_mercury += ax_mercury * dt
    vy_mercury += ay_mercury * dt


    # --------------------------------------------------------
    # Positions
    # --------------------------------------------------------

    sun_x[i + 1] = sun_x[i] + vx_sun * dt
    sun_y[i + 1] = sun_y[i] + vy_sun * dt

    mercury_x[i + 1] = mercury_x[i] + vx_mercury * dt
    mercury_y[i + 1] = mercury_y[i] + vy_mercury * dt


# ============================================================
# 6. CREATE ANIMATION
# ============================================================

fig, ax = plt.subplots(figsize=(8, 8))

# Determine plot limits
max_range = 1.2 * np.max(
    np.sqrt(mercury_x**2 + mercury_y**2)
)

ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)

ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Two-Body Problem: Sun–Mercury")

ax.grid(True)
ax.set_aspect("equal")


# ============================================================
# 7. OBJECTS THAT WILL MOVE
# ============================================================

# Sun
sun_plot, = ax.plot(
    [], [],
    'o',
    markersize=12,
    label="Sun"
)

# Mercury
mercury_plot, = ax.plot(
    [],
    [],
    'o',
    markersize=6,
    label="Mercury"
)

# Mercury orbit trail
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

    sun_plot.set_data([], [])

    mercury_plot.set_data([], [])

    orbit_line.set_data([], [])

    time_text.set_text('')

    return (
        sun_plot,
        mercury_plot,
        orbit_line,
        time_text
    )


# ============================================================
# 9. ANIMATION UPDATE FUNCTION
# ============================================================

def update(frame):

    # Current position of Sun
    sun_plot.set_data(
        [sun_x[frame]],
        [sun_y[frame]]
    )

    # Current position of Mercury
    mercury_plot.set_data(
        [mercury_x[frame]],
        [mercury_y[frame]]
    )

    # Draw Mercury's path
    orbit_line.set_data(
        mercury_x[:frame + 1],
        mercury_y[:frame + 1]
    )

    # Convert seconds → days
    days = time[frame] / (24 * 3600)

    time_text.set_text(
        f"Time = {days:.1f} days"
    )

    return (
        sun_plot,
        mercury_plot,
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