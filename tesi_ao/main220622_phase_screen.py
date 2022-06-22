from arte.atmo import phase_screen_generator
import numpy as np
from tesi_ao.mems_flat_reshaper import MemsFlatReshaper


class FittingErrorPhaseScreen():

    def __init__(self):
        self._mfr = MemsFlatReshaper()
        self._mfr.create_mask(radius=120, center=(231, 306))
        self._mfr.create_reconstructor(set_thresh=0.15)

    def doit(self, r0=0.1):
        self._mfr.apply_phase_screen_distortion(r0)
        self._mfr.do_some_statistic(self._mfr.reference_wavefront(), 3)
        self._wavefront_error = self._mfr.wf_meas[0] - self._mfr.wf_meas[-1]
