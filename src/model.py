import numpy as np
import matplotlib.pyplot as plt


class Nbody:

    def __init__(self, masses, x_pos, y_pos, ux ,uy ,dt , time ,G=1.0):
        self.masses = masses
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.vx = ux
        self.vy = uy
        self.dt = dt
        self.time = time
        self.G = G


    # A cceleration 
    def acceleration(self):
        """
        Calculate the acceleration of each mass due to gravitational forces from all other masses.

        Parameters:
        masses (list): List of masses.
        x_pos (list): List of x positions of the masses.
        y_pos (list): List of y positions of the masses.
        G (float): Gravitational constant.

        Returns:
        ax (list): List of x accelerations for each mass.
        ay (list): List of y accelerations for each mass.
        """
        n = len(self.masses)
        ax = np.zeros(n)
        ay = np.zeros(n)

    

        for i in range(n):

            Fx , Fy = 0 , 0

            for j in range(n):
                if i != j:
                    dx = self.x_pos[j] - self.x_pos[i]
                    dy = self.y_pos[j] - self.y_pos[i]
                    r = np.sqrt(dx**2 + dy**2)
                    Fx += self.G * self.masses[i] * self.masses[j] * dx / (r**3)
                    Fy += self.G * self.masses[i] * self.masses[j] * dy / (r**3)

            ax[i] += Fx / self.masses[i]
            ay[i] += Fy / self.masses[i]

        return ax, ay
    

    # velocity 
    def velocity(self):
        """
        Calculate the velocity of each mass based on its acceleration.

        Returns:
        vx (list): List of x velocities for each mass.
        vy (list): List of y velocities for each mass.
        """
        ax, ay = self.acceleration()
        vx = np.zeros(len(self.masses))
        vy = np.zeros(len(self.masses))

        for i in range(len(self.masses)):
            vx[i] += ax[i] * self.dt + self.ux[i]
            vy[i] += ay[i] * self.dt + self.uy[i]

        return vx, vy



    # position
    def position(self):
        """
        Calculate the position of each mass based on its velocity.

        Returns:
        x_pos (list): List of x positions for each mass.
        y_pos (list): List of y positions for each mass.
        """
        vx, vy = self.velocity()
        x = np.zeros(len(self.masses))
        y = np.zeros(len(self.masses))

        for i in range(len(self.masses)):
            x[i] += vx[i] * self.dt + self.x_pos[i]
            y[i] += vy[i] * self.dt + self.y_pos[i]

        return x,y

