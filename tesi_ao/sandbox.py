
import numpy as np
from plico_interferometer import interferometer
from plico_dm import deformableMirror


def pippo():
    return 44


def pluto():
    pass


def ciao():
    return "ciao!"


def measureCurvatura():
    wyko = interferometer('193.206.155.29', 7300)
    bmc = deformableMirror('193.206.155.92', 7000)

    cmd = np.zeros(140)
    bmc.setShape(cmd)
    wf0 = wyko.wavefront()

    cmd = np.zeros(140)
    cmd[60] = -0.7
    bmc.setShape(cmd)
    dd = wyko.wavefront() - wf0
    wf07 = dd - dd.median()

    cmd = np.zeros(140)
    cmd[60] = -0.2
    bmc.setShape(cmd)
    dd = wyko.wavefront() - wf0
    wf02 = dd - dd.median()


class Robaccia220223(object):

    def __init__(self):
        x = np.array([0, 0.7, 0.8, 0.9])
        y = np.array([0, -1280, -1720, -2220])
        res = np.polyfit(x, y, 2)
        self.a = res[0]
        self.b = res[1]

    def v2p(self, v):
        return self.a * v**2 + self.b * v

    def p2v(self, p):
        v = (-self.b - np.sqrt(self.b**2 + 4 * self.a * p)) / (2 * self.a)
        return v - 0.8
