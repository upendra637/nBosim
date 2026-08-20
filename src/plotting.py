import numpy as np
import matplotlib.pyplot as plt
from model import Nbody

# mass = np.array([1, 0.0005,1e-6])
# x = np.array([0, 0.5, 1.1])
# y = np.array([0, 0, 0])
# ux = np.array([0, 0.2, 0.4])
# uy = np.array([0, 0.7, 1.0])

# masses (solar masses)
masses = [1.0, 1.660e-7, 2.448e-6, 3.003e-6, 3.227e-7,
          9.545e-4, 2.857e-4, 4.366e-5, 5.151e-5]


# average orbital distances from Sun (AU)
X = [0.0, 0.387, 0.723, 1.000, 1.524,
             5.203, 9.537, 19.191, 30.07]

y = [0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0, 0.0, 0.0]

# average orbital velocities (AU/year)
uy = [0.0, 10.07, 7.39, 6.28, 5.09,
              2.62, 2.12, 1.48, 1.14]
ux = [0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0]
dt = 0.01

np.array(masses, dtype=float),
np.array(X, dtype=float),
np.array(y, dtype=float),
np.array(ux, dtype=float),
np.array(uy, dtype=float),

nbody = Nbody(masses, X, y, ux, uy, dt)
N = 10000
x_pos = np.zeros((len(masses), N))
y_pos = np.zeros((len(masses), N))

for i in range(N):
    
    X , Y = nbody.position()
    x_pos[:, i] = X
    y_pos[:, i] = Y

plt.plot(x_pos[0 , :], y_pos[0 , :], label='Mass 1')
plt.plot(x_pos[1 , :], y_pos[1 , :], label='Mass 2')
plt.plot(x_pos[2 , :], y_pos[2 , :], label='Mass 3')
plt.plot(x_pos[3 , :], y_pos[3 , :], label='Mass 4')
plt.plot(x_pos[4 , :], y_pos[4 , :], label='Mass 5')
plt.plot(x_pos[5 , :], y_pos[5 , :], label='Mass 6')
plt.plot(x_pos[6 , :], y_pos[6 , :], label='Mass 7')
plt.plot(x_pos[7 , :], y_pos[7 , :], label='Mass 8')
plt.plot(x_pos[8 , :], y_pos[8 , :], label='Mass 9')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('N-Body Simulation')
plt.legend()
plt.show()