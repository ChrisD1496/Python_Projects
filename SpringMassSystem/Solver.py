import numpy as np
from PyQt5 import QtCore

class ODESolver(QtCore.QObject):

    current_status = QtCore.pyqtSignal(int, int)

    def __init__(self, model, time_step, end_time):
        super().__init__()
        self.calculate_forces = model.calculate_forces
        self.x0 = model.x0
        self.model = model

        self.end_time = end_time
        self.time_step = time_step

        self.reinitialize()

    def reinitialize(self):
        self.n = int(self.end_time / self.time_step)

        self.t = np.linspace(0., self.end_time, self.n)
        self.x = np.zeros((self.n, len(self.x0)))

        self.t[0] = 0.
        self.x[0] = self.x0

    def integrate(self):
        raise NotImplementedError('You need to implement the method integrate')

class Euler(ODESolver):
    def __init__(self, model, time_step, end_time):
        super().__init__(model, time_step, end_time)

    def integrate(self):
        for i in range(0, self.n-1):
            self.t[i+1] = self.t[i] + self.time_step
            self.x[i+1] = self.x[i] + self.time_step * self.calculate_forces(self.t[i], self.x[i])
            self.current_status.emit(i, self.n)

class RungeKutta4(ODESolver):
    def __init__(self, model, time_step, end_time):
        super().__init__(model, time_step, end_time)

    def integrate(self):
        for i in range(0, self.n - 1):
            self.t[i+1] = self.t[i] + self.time_step
            k1 = self.calculate_forces(self.t[i], self.x[i])
            k2 = self.calculate_forces(self.t[i] + self.time_step*0.5, self.x[i] + self.time_step/2 * k1)
            k3 = self.calculate_forces(self.t[i] + self.time_step*0.5, self.x[i] + self.time_step/2 * k2)
            k4 = self.calculate_forces(self.t[i] + self.time_step, self.x[i] + self.time_step * k3)
            self.x[i+1] = self.x[i] + self.time_step * (k1 + 2 * k2 + 2 * k3 + k4) / 6.
            self.current_status.emit(i + 1, self.n - 1)
