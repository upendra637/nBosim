import numpy as np
import matplotlib.pyplot as plt


class Nbody:

    def __init__(self, masses, x_pos, y_pos, ux ,uy ,dt,G=39.478):
        self.masses = masses
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.ux = ux
        self.uy = uy
        self.dt = dt

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
    

        vx = self.ux + ax * self.dt
        vy = self.uy + ay * self.dt


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

        x = self.x_pos + vx * self.dt
        y = self.y_pos + vy * self.dt

        # Update the current position and velocity
        self.x_pos = x
        self.y_pos = y
        self.ux = vx
        self.uy = vy

        return x,y

    def com_frame(self):
        
        """
        Calculate the center of mass frame for the system of masses.

        Returns:
        com_x (float): x position of the center of mass.
        com_y (float): y position of the center of mass.
        com_vx (float): x velocity of the center of mass.
        com_vy (float): y velocity of the center of mass.
        """
        total_mass = np.sum(self.masses)
        com_x = np.sum(self.masses * self.x_pos) / total_mass
        com_y = np.sum(self.masses * self.y_pos) / total_mass

        # com_vx = np.sum(self.masses * self.ux) / total_mass
        # com_vy = np.sum(self.masses * self.uy) / total_mass

        return com_x, com_y


    

