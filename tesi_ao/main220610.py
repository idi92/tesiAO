import numpy as np
from tesi_ao.main220520 import Prove220531
from tesi_ao.zernike_modal_reconstructor import ZernikeToZonal
import matplotlib.pyplot as plt
from arte.utils import modal_decomposer
from arte.types.wavefront import Wavefront



# ifs_fname = "Tesi Edoardo/misure_con_tappo/zifs_pushpull500nmtime0_mcl_def.fits"
# mcl_fname = "Tesi Edoardo/mcl/mcl_all_def.fits"

class Main220610():

    def __init__(self, ifs_fname, mcl_fname):
        
        self.pp = Prove220531(ifs_fname, mcl_fname)
        # pp.create_mask_from_ifs()
        self.pp.create_mask(radius=115, center=(231, 306))
        self.pp.create_reconstructor()
        self.zern2zonal = ZernikeToZonal(self.pp._mzr, self.pp.cmask_obj)
        self.md = modal_decomposer.ModalDecomposer(100)
        self.mask = self.pp.cmask_obj

    def get_wavefront(self):
        return self.pp._wavefront_on_cmask()

    def flatten(self):
        for i in range(10):
            self.pp.one_step()
        self.wf_flat = self.get_wavefront()
        self.cmd_flat = self.pp._bmc.get_shape()



    def apply_zernike(self, n_zern, ampl):
        modal_ampl = np.zeros(50)
        modal_ampl[n_zern-2] = ampl
        pos = self.zern2zonal.convert_modal_to_zonal(modal_ampl)
        self.pp._bmc.set_shape(self.pp._mcl.p2c(pos) + self.cmd_flat)
        self.wf = self.get_wavefront()
        print("std %g - ptp %g - mean %g" % (self.wf.std(), self.wf.ptp(), self.wf.mean()))
        plt.imshow(self.wf)
        plt.colorbar()
        plt.show()

    def decompose_wavefront_on_zernike(self, wf):
        zc = self.md.measureZernikeCoefficientsFromWavefront(Wavefront(wf),
                                                             self.mask)
        plt.plot(zc.zernikeIndexes(), zc.toNumpyArray())
        return zc



def main(ifs_fname, mcl_fname):
    obj = Main220610(ifs_fname, mcl_fname)
    obj.flatten()
    obj.apply_zernike(2, 300e-9)
    obj.decompose_wavefront_on_zernike(obj.wf)
    return obj

