from tesi_ao.main220316 import create_devices, ModeGenerator
from tesi_ao.mems_command_linearization import MemsCommandLinearization
from tesi_ao.mems_command_to_position_linearization_analyzer import CommandToPositionLinearizationAnalyzer
from arte.types.mask import CircularMask
import numpy as np


class Prove2205020():
    
    def __init__(self, mcl_name, cplm_name):
        #mcl_name = '/Users/lbusoni/Downloads/mcl_all_fixedpix.fits'
        #cplm_name = '/Users/lbusoni/Downloads/cplm_mcl_all_fixed-1/cplm_all_fixed.fits'
        self._wyko, self._bmc = create_devices()
        self._mcl = MemsCommandLinearization.load(mcl_name)
        cpla = CommandToPositionLinearizationAnalyzer(cplm_name)
        self._mg = ModeGenerator(cpla, self._mcl)
        
        wf=self._wyko.wavefront() 
        self._mask = CircularMask(wf.shape, maskRadius=120, maskCenter=(235, 310))
        self._mg.THRESHOLD_RMS = 0.1
        self._mg.compute_reconstructor(self._mask)
         
    def spiana(self):
        for i in range(10):
            self.one_step()

    def one_step(self):
        gain = 1
        wf = self._wavefront_on_mask()
        self._dpos = np.zeros(140)
        self._dpos[self._mg._acts_in_pupil] = -1 * gain * np.dot(self._mg._rec, wf.compressed())
        currentpos = self._mcl.c2p(self._bmc.get_shape())
        self._bmc.set_shape(self._mcl.p2c(currentpos+self._dpos))
    
    def _wavefront_on_mask(self):
        wf=self._wyko.wavefront()
        wf=np.ma.array(wf.data, mask=self._mask.mask())
        return wf
    
    def deforma(self):
        pos = np.zeros(140)
        pos[self._mg._acts_in_pupil]=np.random.uniform(
            -1000e-9,1000e-9, self._mg._n_of_selected_acts)
        self._bmc.set_shape(self._mcl.p2c(pos))
    
    
    
    