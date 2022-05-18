import numpy as np
from arte.utils.modal_decomposer import ModalDecomposer
from tesi_ao.mems_command_linearization import MemsCommandLinearization
from arte.utils.zernike_generator import ZernikeGenerator
from random import uniform
from tesi_ao.main220316 import PupilMaskBuilder
from arte.types.wavefront import Wavefront
from arte.types.mask import CircularMask
from future.backports.test.support import time_out


class ShapeReconstructionCommands():
    '''
    the aim of this class is to get new flat reference 
    shape commands for DM, erasing any membrane deformations
    as far as possible 
    '''
    TIME_OUT = 10
    mcl_fname = 'prova/all_act/sandbox/mcl_all_fixedpix.fits'

    def __init__(self, interferometer, mems_deformable_mirror):
        self._interf = interferometer
        self._bmc = mems_deformable_mirror
        self._mcl = MemsCommandLinearization.load(self.mcl_fname)

    def _get_random_cmds(self, act_list=None):
        Nacts = self._bmc.get_number_of_actuators()
        if act_list is None:
            act_list = np.arange(Nacts)
        cmd = np.zeros(Nacts)
        for act in act_list:
            a = np.min(self._mcl._cmd_vector[act])
            b = np.max(self._mcl._cmd_vector[act])
            cmd[act] = uniform(a, b)
        return cmd

    def _get_random_wavefront(self, act_list=None):
        cmd = self._get_random_cmds(act_list)
        self._bmc.set_shape(cmd)
        wf = self._interf.wavefront(timeout_in_sec=self.TIME_OUT)
        return wf

    def _test_decompose_wf(self, mg, act_list=None):

        wf = self._interf.wavefront(timeout_in_sec=self.TIME_OUT)
        wf_mask = wf.mask
        frame_shape = wf.mask.shape
        n_modes_to_decompose = 3

        self._decomposed_wfs = np.ma.zeros(
            (n_modes_to_decompose, frame_shape[0], frame_shape[1]))
        coeff_a = np.zeros(n_modes_to_decompose)
        intersection_mask = np.ma.mask_or(mg._imask, wf_mask)
        pmb = PupilMaskBuilder(intersection_mask)
        yc, xc = pmb.get_barycenter_of_false_pixels()
        D1, D2 = pmb.get_number_of_false_pixels_along_barycenter_axis()
        R_pixels = min(D1, D2) * 0.5 - 2.5
        cmask = CircularMask(frameShape=frame_shape,
                             maskRadius=R_pixels, maskCenter=(yc, xc))
        wf_cmasked = np.ma.array(data=wf.data, mask=cmask.mask())
        decomposer = ModalDecomposer(n_modes_to_decompose)
        # mg._imask = wf_mask
        mg.compute_reconstructor(cmask)
        WF = Wavefront.fromNumpyArray(wf_cmasked)
        # WF_MASK = CircularMask.fromMaskedArray(wf_cmasked)
        coeff_a = decomposer.measureZernikeCoefficientsFromWavefront(
            WF, cmask, n_modes_to_decompose)
        for j in range(2, 2 + n_modes_to_decompose):
            mg.generate_zernike_mode(j, coeff_a[j])
            self._decomposed_wfs[j] = coeff_a[j] * mg._wfmode

    def _get_new_reference_cmds(self, mcl, mg):

        Nacts = self._bmc.get_number_of_actuators()
        cmd0 = np.zeros(Nacts)
        self._bmc.set_shape(cmd0)
        wf_meas = self._interf.wavefront(timeout_in_sec=self.TIME_OUT)
        mg._imask = wf_meas.mask
        mg.compute_reconstructor()
        # compute positions from reconstructor
        pos = np.dot(mg._rec, wf_meas.compressed())
        pos_of_all_acts = np.zeros(Nacts)
        pos_of_all_acts[mg._acts_in_pupil] = pos
        # compute position from bmc cmds
        bmc_cmds = self._bmc.get_shape()
        bmc_pos = np.zeros(Nacts)
        for i in range(Nacts):
            bmc_pos[i] = mcl._finter[i](bmc_cmds[i])
        # compute required cmd
        delta_pos = bmc_pos - pos_of_all_acts
        delta_cmd = np.zeros(Nacts)
        for i in range(Nacts):
            delta_cmd[i] = mcl._sampled_p2c(i, delta_pos[i])
        self._bmc.set_shape(delta_cmd)
        return delta_cmd
