# Solar System in 3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation
from IPython.display import HTML



names = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter",
         "Saturn", "Uranus", "Neptune", "Pluto", "Moon"]

mass = np.array([
    1.0,
    1.6601e-7,   # Mercury
    2.4478e-6,   # Venus
    3.0035e-6,   # Earth (Earth alone, Moon is separate now)
    3.2131e-7,   # Mars
    9.5479e-4,   # Jupiter
    2.8589e-4,   # Saturn
    4.3662e-5,   # Uranus
    5.1514e-5,   # Neptune
    6.5500e-9,   # Pluto
    3.6943e-8,   # Moon
])

# orbital elements for the Sun + 9 planets (starting conditions only)
a = np.array([0.0, 0.3871, 0.7233, 1.0000, 1.5237,
              5.2028, 9.5388, 19.1914, 30.0611, 39.4817])
e = np.array([0.0, 0.2056, 0.0068, 0.0167, 0.0934,
              0.0489, 0.0565, 0.0463, 0.0086, 0.2488])
phase = np.array([0.0, 0.0, 0.7, 1.2, 2.0,
                  2.7, 3.5, 4.0, 5.0, 1.5])
inclination = np.radians([0.0, 7.00, 3.39, 0.00, 1.85,
                          1.30, 2.49, 0.77, 1.77, 17.16])

colors = ["gold", "#b1b1b1", "#e8c39e", "#3f83ff", "#c1440e",
          "#d9a066", "#e3c16f", "#8fd1e0", "#4166f5", "#c9a0dc",
          "lightgray"]

n_bodies = len(names)          # 11
n_planets = 10                 # Sun + 9 planets (the ones with a/e/phase)
sun_idx = names.index("Sun")
earth_idx = names.index("Earth")
moon_idx = names.index("Moon")

G = 4 * np.pi**2
AU_KM = 149_597_870.7


# 2. Starting positions and velocities (3D)

pos0 = np.zeros((n_bodies, 3))
vel0 = np.zeros((n_bodies, 3))
GM_sun = G * mass[sun_idx]

def tilt_about_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]])

# planets: start at perihelion, speed from vis-viva, tilt by inclination
for i in range(1, n_planets):
    r0 = a[i] * (1 - e[i])
    direction = np.array([np.cos(phase[i]), np.sin(phase[i]), 0.0])
    tangent = np.array([-np.sin(phase[i]), np.cos(phase[i]), 0.0])
    speed = np.sqrt(GM_sun * (2 / r0 - 1 / a[i]))

    rot = tilt_about_x(inclination[i])
    pos0[i] = rot @ (r0 * direction)
    vel0[i] = rot @ (speed * tangent)


moon_a_real = 0.00257            # AU (384,400 km)
moon_e = 0.0549
moon_inc = np.radians(5.14)      # tilt relative to the ecliptic
moon_phase = 0.0

r_m = moon_a_real * (1 - moon_e)                       # perigee
GM_em = G * (mass[earth_idx] + mass[moon_idx])
v_m = np.sqrt(GM_em * (2 / r_m - 1 / moon_a_real))

rot_m = tilt_about_x(moon_inc)
rel_pos = rot_m @ (r_m * np.array([np.cos(moon_phase), np.sin(moon_phase), 0.0]))
rel_vel = rot_m @ (v_m * np.array([-np.sin(moon_phase), np.cos(moon_phase), 0.0]))

m_e, m_m = mass[earth_idx], mass[moon_idx]
M_em = m_e + m_m
bary_pos = pos0[earth_idx].copy()
bary_vel = vel0[earth_idx].copy()

pos0[earth_idx] = bary_pos - (m_m / M_em) * rel_pos
vel0[earth_idx] = bary_vel - (m_m / M_em) * rel_vel
pos0[moon_idx]  = bary_pos + (m_e / M_em) * rel_pos
vel0[moon_idx]  = bary_vel + (m_e / M_em) * rel_vel


total_momentum = np.sum(mass[1:, None] * vel0[1:], axis=0)
vel0[sun_idx] = -total_momentum / mass[sun_idx]


def acceleration(pos):
    diff = pos[None, :, :] - pos[:, None, :]      # diff[i, j] = pos[j] - pos[i]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)                # avoid 0-division on self
    inv_d3 = dist_sq ** -1.5
    np.fill_diagonal(inv_d3, 0.0)                 # a body doesn't pull on itself
    return G * np.einsum("ij,ijk->ik", inv_d3 * mass[None, :], diff)


# 4. RK4 step

def rk4_step(pos, vel, dt):
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


simulation_years = 165
frames = 1500
substeps_per_frame = 80                       # dt ~ 0.5 day (needed for the Moon)
total_steps = frames * substeps_per_frame
dt = simulation_years / total_steps
print(f"dt = {dt * 365.25:.3f} days, {total_steps:,} RK4 steps (this may take a minute)...")

pos = pos0.copy()
vel = vel0.copy()
trajectory = np.zeros((frames, n_bodies, 3))

frame_i = 0
for step in range(total_steps):
    pos, vel = rk4_step(pos, vel, dt)
    if (step + 1) % substeps_per_frame == 0:
        trajectory[frame_i] = pos
        frame_i += 1


moon_rel_true = trajectory[:, moon_idx] - trajectory[:, earth_idx]
moon_dist_km = np.linalg.norm(moon_rel_true, axis=1) * AU_KM
print(f"Earth-Moon distance over run: {moon_dist_km.min():,.0f} to {moon_dist_km.max():,.0f} km")


moon_a_display = 0.35                          # AU, drawn radius
moon_scale = moon_a_display / moon_a_real      # ~136x

plot_traj = trajectory.copy()
plot_traj[:, moon_idx] = trajectory[:, earth_idx] + moon_scale * moon_rel_true

body_x = plot_traj[:, :, 0].T
body_y = plot_traj[:, :, 1].T
body_z = plot_traj[:, :, 2].T


# 7. Figure setup (3D)

fig = plt.figure(figsize=(11, 11))
fig.patch.set_facecolor("black")
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("black")

lim = 42
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)
ax.set_box_aspect((1, 1, 1))

ax.set_xlabel("X (AU)", color="white")
ax.set_ylabel("Y (AU)", color="white")
ax.set_zlabel("Z (AU)", color="white")
ax.set_title("Solar System (RK4 N-body, 3D) with the Moon as a true N-body", color="white")
ax.tick_params(colors="white")

for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.set_facecolor((0, 0, 0, 1.0))
    pane.set_edgecolor("gray")
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis._axinfo["grid"]["color"] = (0.3, 0.3, 0.3, 0.3)

ax.view_init(elev=25, azim=45)


for i in range(n_bodies):
    if i == moon_idx:
        continue
    ax.plot(body_x[i], body_y[i], body_z[i], linestyle="--", alpha=0.25, color=colors[i])

body_points = []
for i in range(n_bodies):
    if i == earth_idx:
        p, = ax.plot([], [], [], "o", markersize=10, color=colors[i],
                     markeredgecolor="white", markeredgewidth=1.5,
                     label=names[i], zorder=5)
    elif i == sun_idx:
        p, = ax.plot([], [], [], "o", markersize=14, color=colors[i], label=names[i])
    elif i == moon_idx:
        p, = ax.plot([], [], [], "o", markersize=4, color=colors[i],
                     markeredgecolor="white", label=f"Moon (offset x{moon_scale:.0f})",
                     zorder=5)
    else:
        p, = ax.plot([], [], [], "o", markersize=5, color=colors[i], label=names[i])
    body_points.append(p)


moon_orbit_line, = ax.plot([], [], [], linestyle=":", alpha=0.4, color="lightgray")

ax.legend(loc="upper right", fontsize=8, ncol=2, facecolor="black", edgecolor="gray", labelcolor="white")

theta_ring = np.linspace(0, 2 * np.pi, 150)
earth_x, earth_y, earth_z = body_x[earth_idx], body_y[earth_idx], body_z[earth_idx]

def update(frame):
    for i in range(n_bodies):
        body_points[i].set_data_3d([body_x[i, frame]], [body_y[i, frame]], [body_z[i, frame]])

    ex, ey, ez = earth_x[frame], earth_y[frame], earth_z[frame]
    ring_x = ex + moon_a_display * np.cos(theta_ring)
    ring_y = ey + moon_a_display * np.sin(theta_ring) * np.cos(moon_inc)
    ring_z = ez + moon_a_display * np.sin(theta_ring) * np.sin(moon_inc)
    moon_orbit_line.set_data_3d(ring_x, ring_y, ring_z)

    return body_points + [moon_orbit_line]


animation = FuncAnimation(fig, update, frames=frames, interval=15, blit=False)
plt.close()

HTML(animation.to_jshtml())