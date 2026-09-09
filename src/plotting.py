import numpy as np
import matplotlib.pyplot as plt
from model import Nbody


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
    5.151e-5]
)


# average orbital distances from Sun (AU)
X = np.array(
    [0.0, 0.387, 0.723, 1.000, 1.524,5.203, 9.537, 19.191, 30.07]
    )

y = np.zeros_like(X)  # All planets start on the x-axis

# average orbital velocities (AU/year)
uy = np.array(
    [0.0, 10.07, 7.39, 6.28, 5.09,2.62, 2.12, 1.48, 1.14]
    )

ux = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0]
    )

dt = 0.01
nbody = Nbody(masses, X, y, ux, uy, dt)
N = 10000

x_pos = np.zeros((len(masses), N))
y_pos = np.zeros((len(masses), N))

for i in range(N):
    
    X , Y = nbody.position()
    x_pos[:, i] = X
    y_pos[:, i] = Y

X_xom = np.zeros_like(x_pos)
Y_xom = np.zeros_like(y_pos)

com_x, com_y = nbody.com_frame()

for i in range(N):
    
    X_xom[:, i] = x_pos[:, i] - com_x
    Y_xom[:, i] = y_pos[:, i] - com_y

labels = [
    "Sun",
    "Mercury",
    "Venus",
    "Earth",
    "Mars",
    "Jupiter",
    "Saturn",
    "Uranus",
    "Neptune"
]

# for i in range(len(masses)):
#     plt.plot(x_pos[i, :], y_pos[i, :], label=labels[i])

for i in range(len(masses)):
    plt.plot(X_xom[i, :], Y_xom[i, :], label=labels[i])

plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('N-Body Simulation')
plt.legend()
plt.show()