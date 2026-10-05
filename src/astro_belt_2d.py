import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ============================================================
# 1. CONSTANTS AND UNITS
# ============================================================
#
# Distance = AU
# Time     = years
# Mass     = Solar masses
#
# In these units:
#
# G = 4*pi^2
#
# ============================================================

G = 4 * np.pi**2


# ============================================================
# 2. SIMULATION PARAMETERS
# ============================================================

N_ASTEROIDS = 1000

T_MAX = 12.0       # years
DT = 0.002         # years

N_STEPS = int(T_MAX / DT)


# ============================================================
# 3. PLANET DATA
# ============================================================

# Masses in Solar masses

masses = np.array([
    1.0,             # Sun
    1.660e-7,        # Mercury
    2.447e-6,        # Venus
    3.003e-6,        # Earth
    3.227e-7,        # Mars
    9.545e-4         # Jupiter
])


names = [
    "Sun",
    "Mercury",
    "Venus",
    "Earth",
    "Mars",
    "Jupiter"
]


# Semi-major axes in AU

a = np.array([
    0.0,
    0.387,
    0.723,
    1.000,
    1.524,
    5.203
])


# ============================================================
# 4. INITIAL PLANET POSITIONS AND VELOCITIES
# ============================================================

N_PLANETS = len(masses)

planet_pos = np.zeros(
    (N_PLANETS, 2)
)

planet_vel = np.zeros(
    (N_PLANETS, 2)
)


# ------------------------------------------------------------
# Sun
# ------------------------------------------------------------

planet_pos[0] = [0.0, 0.0]

planet_vel[0] = [0.0, 0.0]


# ------------------------------------------------------------
# Planets
# ------------------------------------------------------------
#
# Initially place each planet on the x-axis.
#
# Velocity is perpendicular to the radius vector.
#
# This produces approximately circular orbits.
#
# ------------------------------------------------------------

for i in range(1, N_PLANETS):

    r = a[i]

    planet_pos[i] = [r, 0.0]

    # Circular orbital velocity
    v = np.sqrt(
        G * masses[0] / r
    )

    # Tangential velocity
    planet_vel[i] = [0.0, v]


# ============================================================
# 5. CREATE ASTEROIDS
# ============================================================

asteroid_pos = np.zeros(
    (N_ASTEROIDS, 2)
)

asteroid_vel = np.zeros(
    (N_ASTEROIDS, 2)
)


# Random number generator

rng = np.random.default_rng(42)


# ------------------------------------------------------------
# Generate asteroid belt
# ------------------------------------------------------------

for i in range(N_ASTEROIDS):

    # --------------------------------------------------------
    # Semi-major axis / orbital radius
    #
    # Main asteroid belt:
    #
    # 2.1 AU ---> 3.3 AU
    # --------------------------------------------------------

    r = rng.uniform(
        2.1,
        3.3
    )


    # --------------------------------------------------------
    # Random angular position
    # --------------------------------------------------------

    theta = rng.uniform(
        0,
        2 * np.pi
    )


    # --------------------------------------------------------
    # Position
    # --------------------------------------------------------

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    asteroid_pos[i] = [
        x,
        y
    ]


    # --------------------------------------------------------
    # Circular orbital velocity
    # --------------------------------------------------------

    v = np.sqrt(
        G * masses[0] / r
    )


    # --------------------------------------------------------
    # Tangential velocity
    #
    # vx = -v sin(theta)
    # vy =  v cos(theta)
    #
    # --------------------------------------------------------

    vx = -v * np.sin(theta)
    vy =  v * np.cos(theta)

    asteroid_vel[i] = [
        vx,
        vy
    ]


# ============================================================
# 6. PLANET ACCELERATION
# ============================================================

def calculate_planet_acceleration(pos):

    """
    Calculate gravitational acceleration of all planets.

    pos shape:
        (N_PLANETS, 2)
    """

    n = len(pos)

    acceleration = np.zeros_like(pos)


    for i in range(n):

        for j in range(n):

            # A body does not interact with itself
            if i == j:
                continue


            dx = (
                pos[j, 0]
                - pos[i, 0]
            )

            dy = (
                pos[j, 1]
                - pos[i, 1]
            )


            distance_squared = (
                dx**2 + dy**2
            )

            distance = np.sqrt(
                distance_squared
            )


            acceleration[i] += (
                G
                * masses[j]
                * np.array([dx, dy])
                / distance**3
            )


    return acceleration


# ============================================================
# 7. ASTEROID ACCELERATION
# ============================================================

def calculate_asteroid_acceleration(
    asteroid_pos,
    planet_pos
):

    """
    Calculate gravitational acceleration on every asteroid.

    The asteroids feel gravity from:
        Sun
        Mercury
        Venus
        Earth
        Mars
        Jupiter

    Asteroid-asteroid gravity is ignored because asteroid
    masses are extremely small.

    This is the restricted N-body approximation.
    """


    acceleration = np.zeros_like(
        asteroid_pos
    )


    # --------------------------------------------------------
    # Loop over Sun + planets
    # --------------------------------------------------------

    for j in range(
        len(planet_pos)
    ):

        # Difference:
        #
        # planet position - asteroid position
        #
        dx = (
            planet_pos[j, 0]
            - asteroid_pos[:, 0]
        )

        dy = (
            planet_pos[j, 1]
            - asteroid_pos[:, 1]
        )


        distance_squared = (
            dx**2 + dy**2
        )

        distance = np.sqrt(
            distance_squared
        )


        acceleration[:, 0] += (
            G
            * masses[j]
            * dx
            / distance**3
        )


        acceleration[:, 1] += (
            G
            * masses[j]
            * dy
            / distance**3
        )


    return acceleration


# ============================================================
# 8. DERIVATIVES FOR THE COMPLETE SYSTEM
# ============================================================

def derivatives(
    planet_pos,
    planet_vel,
    asteroid_pos,
    asteroid_vel
):

    # Planet acceleration
    planet_acc = (
        calculate_planet_acceleration(
            planet_pos
        )
    )


    # Asteroid acceleration
    asteroid_acc = (
        calculate_asteroid_acceleration(
            asteroid_pos,
            planet_pos
        )
    )


    # Equations:
    #
    # dr/dt = v
    # dv/dt = a

    return (
        planet_vel,
        planet_acc,
        asteroid_vel,
        asteroid_acc
    )


# ============================================================
# 9. RK4 INTEGRATOR
# ============================================================

def rk4_step(
    planet_pos,
    planet_vel,
    asteroid_pos,
    asteroid_vel,
    dt
):

    # ========================================================
    # K1
    # ========================================================

    k1_p, k1_v, k1_ap, k1_av = derivatives(
        planet_pos,
        planet_vel,
        asteroid_pos,
        asteroid_vel
    )


    # ========================================================
    # K2
    # ========================================================

    k2_p, k2_v, k2_ap, k2_av = derivatives(
        planet_pos + 0.5 * dt * k1_p,
        planet_vel + 0.5 * dt * k1_v,
        asteroid_pos + 0.5 * dt * k1_ap,
        asteroid_vel + 0.5 * dt * k1_av
    )


    # ========================================================
    # K3
    # ========================================================

    k3_p, k3_v, k3_ap, k3_av = derivatives(
        planet_pos + 0.5 * dt * k2_p,
        planet_vel + 0.5 * dt * k2_v,
        asteroid_pos + 0.5 * dt * k2_ap,
        asteroid_vel + 0.5 * dt * k2_av
    )


    # ========================================================
    # K4
    # ========================================================

    k4_p, k4_v, k4_ap, k4_av = derivatives(
        planet_pos + dt * k3_p,
        planet_vel + dt * k3_v,
        asteroid_pos + dt * k3_ap,
        asteroid_vel + dt * k3_av
    )


    # ========================================================
    # RK4 UPDATE: PLANETS
    # ========================================================

    new_planet_pos = planet_pos + (dt / 6.0) * (
        k1_p
        + 2.0 * k2_p
        + 2.0 * k3_p
        + k4_p
    )


    new_planet_vel = planet_vel + (dt / 6.0) * (
        k1_v
        + 2.0 * k2_v
        + 2.0 * k3_v
        + k4_v
    )


    # ========================================================
    # RK4 UPDATE: ASTEROIDS
    # ========================================================

    new_asteroid_pos = asteroid_pos + (dt / 6.0) * (
        k1_ap
        + 2.0 * k2_ap
        + 2.0 * k3_ap
        + k4_ap
    )


    new_asteroid_vel = asteroid_vel + (dt / 6.0) * (
        k1_av
        + 2.0 * k2_av
        + 2.0 * k3_av
        + k4_av
    )


    return (
        new_planet_pos,
        new_planet_vel,
        new_asteroid_pos,
        new_asteroid_vel
    )


    # ========================================================
    # UPDATE PLANET POSITIONS
    # ========================================================

    new_planet_pos = (
        planet_pos
        + (dt / 6.0)
        * (
            k1_p
            + 2*k2_p
            + 2*k3_p
            + k4_p
        )
    )


    # ========================================================
    # UPDATE PLANET VELOCITIES
    # ========================================================

    new_planet_vel = (
        planet_vel
        + (dt / 6.0)
        * (
            k1_v
            + 2*k2_v
            + 2*k3_v
            + k4_v
        )
    )


    # ========================================================
    # UPDATE ASTEROID POSITIONS
    # ========================================================

    new_asteroid_pos = (
        asteroid_pos
        + (dt / 6.0)
        * (
            k1_ap
            + 2*k2_ap
            + 2*k3_ap
            + k4_ap
        )
    )


    # ========================================================
    # UPDATE ASTEROID VELOCITIES
    # ========================================================

    new_asteroid_vel = (
        asteroid_vel
        + (dt / 6.0)
        * (
            k1_av
            + 2*k2_av
            + 2*k3_av
            + k4_av
        )
    )


    return (
        new_planet_pos,
        new_planet_vel,
        new_asteroid_pos,
        new_asteroid_vel
    )


# ============================================================
# 10. HISTORY ARRAYS
# ============================================================

planet_history = np.zeros(
    (
        N_STEPS + 1,
        N_PLANETS,
        2
    )
)


asteroid_history = np.zeros(
    (
        N_STEPS + 1,
        N_ASTEROIDS,
        2
    )
)


# ============================================================
# 11. STORE INITIAL CONDITIONS
# ============================================================

planet_history[0] = planet_pos

asteroid_history[0] = asteroid_pos


# ============================================================
# 12. RUN SIMULATION
# ============================================================

print()
print("========================================")
print("Starting RK4 Solar System Simulation")
print("========================================")
print()

print(
    f"Number of asteroids : {N_ASTEROIDS}"
)

print(
    f"Simulation time     : {T_MAX} years"
)

print(
    f"Time step           : {DT} years"
)

print()


for step in range(N_STEPS):

    (
        planet_pos,
        planet_vel,
        asteroid_pos,
        asteroid_vel
    ) = rk4_step(

        planet_pos,
        planet_vel,

        asteroid_pos,
        asteroid_vel,

        DT
    )


    # Store positions
    planet_history[step + 1] = (
        planet_pos
    )

    asteroid_history[step + 1] = (
        asteroid_pos
    )


    # Progress
    if step % 500 == 0:

        progress = (
            100
            * step
            / N_STEPS
        )

        print(
            f"Progress: {progress:.1f}%"
        )


print()
print("Simulation complete.")
print()


# ============================================================
# 13. STATIC ASTEROID-BELT PLOT
# ============================================================

plt.figure(
    figsize=(10, 10)
)


# ------------------------------------------------------------
# Asteroids
#
# IMPORTANT:
# Only plot ONE time frame.
# Otherwise millions of points overlap.
# ------------------------------------------------------------

plt.scatter(

    asteroid_history[-1, :, 0],

    asteroid_history[-1, :, 1],

    s=2,

    alpha=0.7,

    label="Asteroids"
)


# ------------------------------------------------------------
# Planetary orbits
# ------------------------------------------------------------

for i in range(
    1,
    N_PLANETS
):

    plt.plot(

        planet_history[:, i, 0],

        planet_history[:, i, 1],

        label=names[i]
    )


# ------------------------------------------------------------
# Sun
# ------------------------------------------------------------

plt.scatter(

    0,
    0,

    s=200,

    label="Sun"
)


# ------------------------------------------------------------
# Plot settings
# ------------------------------------------------------------

plt.xlabel(
    "x (AU)"
)

plt.ylabel(
    "y (AU)"
)

plt.title(
    "2D Solar System with Asteroid Belt — RK4"
)

plt.axis("equal")

plt.xlim(
    -6,
    6
)

plt.ylim(
    -6,
    6
)

plt.legend()

plt.grid(
    alpha=0.2
)

plt.show()


# ============================================================
# 14. ANIMATION
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 10)
)


# ------------------------------------------------------------
# Axis limits
# ------------------------------------------------------------

ax.set_xlim(
    -6,
    6
)

ax.set_ylim(
    -6,
    6
)


ax.set_aspect(
    "equal"
)


ax.set_xlabel(
    "x (AU)"
)

ax.set_ylabel(
    "y (AU)"
)

ax.set_title(
    "2D Solar System — RK4 Asteroid-Belt Simulation"
)


# ============================================================
# 15. SUN
# ============================================================

sun_plot, = ax.plot(

    [],
    [],

    marker="o",

    markersize=10,

    linestyle=""
)


# ============================================================
# 16. PLANETS
# ============================================================

planet_plots = []


for i in range(
    1,
    N_PLANETS
):

    p, = ax.plot(

        [],
        [],

        marker="o",

        markersize=5,

        linestyle=""
    )

    planet_plots.append(p)




asteroid_plot = ax.scatter(

    [],

    [],

    s=2,

    alpha=0.7
)




time_text = ax.text(

    0.02,
    0.95,

    "",

    transform=ax.transAxes
)




def init():

    # Sun
    sun_plot.set_data(
        [],
        []
    )


    # Planets
    for p in planet_plots:

        p.set_data(
            [],
            []
        )


    # Asteroids
    asteroid_plot.set_offsets(
        np.empty(
            (0, 2)
        )
    )


    time_text.set_text(
        ""
    )


    return (
        [sun_plot]
        + planet_plots
        + [asteroid_plot]
        + [time_text]
    )




def update(frame):

    # --------------------------------------------------------
    # Current simulation time
    # --------------------------------------------------------

    current_time = (
        frame * DT
    )


  

    sun_plot.set_data(

        [planet_history[
            frame,
            0,
            0
        ]],

        [planet_history[
            frame,
            0,
            1
        ]]
    )


   

    for i, p in enumerate(
        planet_plots
    ):

        planet_index = i + 1


        p.set_data(

            [planet_history[
                frame,
                planet_index,
                0
            ]],

            [planet_history[
                frame,
                planet_index,
                1
            ]]
        )


    

    asteroid_plot.set_offsets(

        asteroid_history[
            frame
        ]
    )



    time_text.set_text(

        f"Time = {current_time:.2f} years"
    )


    return (
        [sun_plot]
        + planet_plots
        + [asteroid_plot]
        + [time_text]
    )


# ============================================================
# 21. CREATE ANIMATION
# ============================================================

ani = FuncAnimation(

    fig,

    update,

    frames=N_STEPS + 1,

    init_func=init,

    interval=20,

    blit=True
)




plt.show()







from IPython.display import HTML, display
from matplotlib.animation import FuncAnimation

# ============================================================
# ANIMATION WITH ORBIT TRAILS
# ============================================================

fig, ax = plt.subplots(figsize=(10, 10))

ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)

ax.set_aspect("equal")

ax.set_xlabel("x (AU)")
ax.set_ylabel("y (AU)")

ax.set_title(
    "2D Solar System — RK4 Asteroid-Belt Simulation"
)


# ============================================================
# 1. DRAW THE COMPLETE PLANETARY ORBITS
# ============================================================
#
# These are the trajectories calculated by the RK4 simulation.
#
# They are drawn in the background.
#
# ============================================================

orbit_plots = []

for i in range(1, N_PLANETS):

    orbit, = ax.plot(
        planet_history[:, i, 0],
        planet_history[:, i, 1],
        linewidth=1,
        alpha=0.6,
        label=names[i]
    )

    orbit_plots.append(orbit)


# ============================================================
# 2. SUN
# ============================================================

sun_plot, = ax.plot(
    [],
    [],
    marker="o",
    markersize=10,
    linestyle=""
)


# ============================================================
# 3. PLANET MARKERS
# ============================================================

planet_plots = []

for i in range(1, N_PLANETS):

    p, = ax.plot(
        [],
        [],
        marker="o",
        markersize=6,
        linestyle=""
    )

    planet_plots.append(p)


# ============================================================
# 4. ASTEROIDS
# ============================================================

asteroid_plot = ax.scatter(
    [],
    [],
    s=2,
    alpha=0.7
)


# ============================================================
# 5. TIME DISPLAY
# ============================================================

time_text = ax.text(
    0.02,
    0.95,
    "",
    transform=ax.transAxes
)


# ============================================================
# 6. INITIALIZATION
# ============================================================

def init():

    # Sun
    sun_plot.set_data(
        [],
        []
    )

    # Planets
    for p in planet_plots:

        p.set_data(
            [],
            []
        )

    # Asteroids
    asteroid_plot.set_offsets(
        np.empty((0, 2))
    )

    # Time
    time_text.set_text("")

    return (
        [sun_plot]
        + planet_plots
        + [asteroid_plot]
        + [time_text]
    )


# ============================================================
# 7. UPDATE FUNCTION
# ============================================================

def update(frame):

    # --------------------------------------------------------
    # Current simulation time
    # --------------------------------------------------------

    current_time = frame * DT


    # --------------------------------------------------------
    # SUN
    # --------------------------------------------------------

    sun_plot.set_data(

        [planet_history[
            frame,
            0,
            0
        ]],

        [planet_history[
            frame,
            0,
            1
        ]]
    )


    # --------------------------------------------------------
    # PLANETS
    # --------------------------------------------------------

    for i, p in enumerate(
        planet_plots
    ):

        planet_index = i + 1

        p.set_data(

            [planet_history[
                frame,
                planet_index,
                0
            ]],

            [planet_history[
                frame,
                planet_index,
                1
            ]]
        )


    # --------------------------------------------------------
    # ASTEROIDS
    # --------------------------------------------------------

    asteroid_plot.set_offsets(
        asteroid_history[frame]
    )


    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    time_text.set_text(
        f"Time = {current_time:.2f} years"
    )


    return (
        [sun_plot]
        + planet_plots
        + [asteroid_plot]
        + [time_text]
    )


# ============================================================
# 8. ANIMATION FRAMES
# ============================================================

animation_frames = range(
    0,
    N_STEPS + 1,
    10
)


# ============================================================
# 9. CREATE ANIMATION
# ============================================================

ani = FuncAnimation(

    fig,

    update,

    frames=animation_frames,

    init_func=init,

    interval=30,

    blit=True
)


# ============================================================
# 10. DISPLAY IN JUPYTER
# ============================================================

plt.close(fig)

display(
    HTML(
        ani.to_jshtml()
    )
)





