from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sys
import SpringMassSystem
import Solver

class GUI(QtWidgets.QWidget):
    def __init__(self):
        super(GUI, self).__init__()

        self.mythread = None
        self.solver = None
        self.model = None

        self.title = "Final Project"

        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(self.main_layout)

        self.tabs = QtWidgets.QTabWidget()
        self.main_layout.addWidget(self.tabs)

        self.tab1 = QtWidgets.QWidget()
        self.tab2 = QtWidgets.QWidget()
        self.tab3 = QtWidgets.QWidget()

        self.tabs.addTab(self.tab1, "Model Settings")
        self.tabs.addTab(self.tab2, "Solver Settings")
        self.tabs.addTab(self.tab3, "Charts")

        self.tab1_layout = QtWidgets.QVBoxLayout(self)
        self.tab1.setLayout(self.tab1_layout)

        self.tab2_layout = QtWidgets.QVBoxLayout(self)
        self.tab2.setLayout(self.tab2_layout)

        self.tab3_layout = QtWidgets.QVBoxLayout(self)
        self.tab3.setLayout(self.tab3_layout)

        # First Tab Setting

        combo_frame_settings = QtWidgets.QGroupBox()
        combo_frame_settings.setTitle('Settings')
        self.combo_solver = QtWidgets.QComboBox()
        self.combo_model = QtWidgets.QComboBox()
        self.combo_model.addItems(['Damped Spring-Mass System', 'Undamped Spring-Mass System'])
        self.combo_model.activated.connect(self.spring_system_type)
        self.combo_solver.addItems(['Euler', 'Runge Kutta'])
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.combo_model)
        vbox.addWidget(self.combo_solver)
        vbox.addStretch(1)
        combo_frame_settings.setLayout(vbox)

        combo_frame_calculation = QtWidgets.QGroupBox()
        combo_frame_calculation.setTitle('Calculation')
        self.button = QtWidgets.QPushButton('Calculate')
        self.button.clicked.connect(self.calculate)
        self.label = QtWidgets.QLabel(self)
        self.label.setText("")
        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.button)
        vbox.addWidget(self.label)
        vbox.addWidget(self.progress_bar)
        vbox.addStretch(1)
        combo_frame_calculation.setLayout(vbox)

        combo_frame_start_conditions = QtWidgets.QGroupBox()
        combo_frame_start_conditions.setTitle('Initial Conditions')
        self.label_x0 = QtWidgets.QLabel(self)
        self.label_x0.setText("Initial Height:")
        self.line_x0 = QtWidgets.QLineEdit()
        self.line_x0.setText("5")
        self.label_v0 = QtWidgets.QLabel(self)
        self.label_v0.setText("Initial Velocity:")
        self.line_v0 = QtWidgets.QLineEdit()
        self.line_v0.setText("0")
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.label_x0)
        vbox.addWidget(self.line_x0)
        vbox.addWidget(self.label_v0)
        vbox.addWidget(self.line_v0)
        vbox.addStretch(1)
        combo_frame_start_conditions.setLayout(vbox)

        combo_frame_model_setting = QtWidgets.QGroupBox()
        combo_frame_model_setting.setTitle('Model Setting')
        self.label_m = QtWidgets.QLabel(self)
        self.label_m.setText("Mass of the System:")
        self.line_m = QtWidgets.QLineEdit()
        self.line_m.setText("2")
        self.label_b = QtWidgets.QLabel(self)
        self.label_b.setText("Damping of the System:")
        self.line_b = QtWidgets.QLineEdit()
        self.line_b.setText("0.1")
        self.label_c = QtWidgets.QLabel(self)
        self.label_c.setText("Spring Constant of the System:")
        self.line_c = QtWidgets.QLineEdit()
        self.line_c.setText("0.5")
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.label_m)
        vbox.addWidget(self.line_m)
        vbox.addWidget(self.label_b)
        vbox.addWidget(self.line_b)
        vbox.addWidget(self.label_c)
        vbox.addWidget(self.line_c)
        vbox.addStretch(1)
        combo_frame_model_setting.setLayout(vbox)

        spacer = QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.tab1_layout.addWidget(combo_frame_settings)
        self.tab1_layout.addSpacerItem(spacer)
        self.tab1_layout.addWidget(combo_frame_model_setting)
        self.tab1_layout.addSpacerItem(spacer)
        self.tab1_layout.addWidget(combo_frame_start_conditions)
        self.tab1_layout.addSpacerItem(spacer)
        self.tab1_layout.addWidget(combo_frame_calculation)

        # Second Tab Settings

        combo_frame_config = QtWidgets.QGroupBox()
        combo_frame_config.setTitle('Configuration')
        self.label_dt = QtWidgets.QLabel(self)
        self.label_dt.setText("Solver Time Step (dt):")
        self.line_dt = QtWidgets.QLineEdit()
        self.line_dt.setText("0.1")
        self.label_t_end = QtWidgets.QLabel(self)
        self.label_t_end.setText("Solver End Time:")
        self.line_t_end = QtWidgets.QLineEdit()
        self.line_t_end.setText("50")
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.label_dt)
        vbox.addWidget(self.line_dt)
        vbox.addWidget(self.label_t_end)
        vbox.addWidget(self.line_t_end)
        vbox.addStretch(1)
        combo_frame_config.setLayout(vbox)

        self.tab2_layout.addWidget(combo_frame_config)

        # Third Tab Settings

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.tab3_layout.addWidget(self.canvas)

        self.resize(600, 800)

    def spring_system_type(self):
        self.model_name = self.combo_model.currentText()
        if self.model_name == 'Undamped Spring-Mass System':
            self.line_b.setDisabled(True)
        else:
            self.line_b.setDisabled(False)

    def evaluate_line_edits(self):
        try:
            dt = float(self.line_dt.text())
            t_end = float(self.line_t_end.text())
            m = float(self.line_m.text())
            b = 0 if self.model_name == 'Undamped Spring-Mass System' else float(self.line_b.text())
            c = float(self.line_c.text())
            x0 = float(self.line_x0.text())
            v0 = float(self.line_v0.text())

        except ValueError:
            QtWidgets.QMessageBox.critical(self, 'Value Error', 'One of the Values is not a number')
            return None, None, None, None, None, None, None

        if not dt < t_end:
            QtWidgets.QMessageBox.critical(self, 'Settings Error', 'End time must be greater than time step!')
            return None, None

        return dt, t_end, m, b, c, x0, v0

    def calculate(self):
        self.button.setEnabled(False)
        self.model_name = self.combo_model.currentText()
        self.solver_name = self.combo_solver.currentText()

        dt, t_end, m, b, c, x0, v0 = self.evaluate_line_edits()

        if not dt:
            self.button.setEnabled(True)
            return

        self.model = SpringMassSystem.SpringMassSystem(m, b, c, x0, v0)

        if self.solver_name == 'Euler':
            self.solver = Solver.Euler(self.model, dt, t_end)
        else:
            self.solver = Solver.RungeKutta4(self.model, dt, t_end)
        self.solver.current_status.connect(self.update_progress)

        self.solver.reinitialize()

        self.mythread = MyThread(self.solver)
        self.mythread.thread_finished.connect(self.plot)
        self.mythread.thread_finished.connect(self.update_progress)
        self.mythread.start()

    def plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.clear()
        ax.plot(self.solver.t, self.solver.x[:, 0], 'b-')
        ax.set_xlabel('Time in [s]')
        ax.set_ylabel('Displacement in [m]')
        ax.set_title('Output ' + self.model_name + ' ' + self.solver_name)

        ax2 = ax.twinx()
        ax2.set_ylabel('Velocity in [m/s]')

        ax.plot(self.solver.t, self.solver.x[:, 1], 'r--')
        ax.legend(['Displacement', 'Velocity'], loc='best')

        self.canvas.draw()
        self.button.setEnabled(True)

    def update_progress(self, current, final=0):
        if final > 0:
            progress = current / final * 100
        else:
            progress = 100
        self.label.setText("Progress: %4.1f" % progress)
        self.progress_bar.setValue(int(progress))


class MyThread(QtCore.QThread):
    thread_finished = QtCore.pyqtSignal(int)

    def __init__(self, solver):
        super().__init__()
        self.solver = solver

    def run(self):
        self.solver.integrate()
        self.thread_finished.emit(100)

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    gui = GUI()
    gui.show()
    ret = app.exec_()
    sys.exit(ret)
