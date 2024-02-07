import matplotlib.pyplot as plt
import numpy as np
from Solver import Euler


class SpringMassSystem:
    def __init__(self, m, b, c, x0, v0):
        self.mass = m
        self.damping_coefficient = b
        self.spring_constant = c
        self.x0 = np.array([x0, v0])

    def calculate_forces(self, time, state):
        position_derivative = state[1]
        velocity_derivative = -(self.damping_coefficient / self.mass) * state[1] - (self.spring_constant / self.mass) * \
                              state[0]
        return np.array([position_derivative, velocity_derivative])


if __name__ == "__main__":
    spring_mass_system = SpringMassSystem(m=1.0, b=0.5, c=2.0, x0=0.0, v0=1.0)

    euler_solver = Euler(spring_mass_system)
    euler_solver.integrate()

    plt.plot(euler_solver.t, euler_solver.x[:, 0], label="Position")
    plt.plot(euler_solver.t, euler_solver.x[:, 1], label="Velocity")
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.legend()
    plt.show()
