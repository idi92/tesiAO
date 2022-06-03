from tesi_ao.main220316 import create_devices, ModeGenerator, PupilMaskBuilder
from tesi_ao.mems_command_linearization import MemsCommandLinearization
from tesi_ao.mems_command_to_position_linearization_analyzer import CommandToPositionLinearizationAnalyzer
from arte.types.mask import CircularMask
from tesi_ao.mems_zonal_influence_functions_measurer import ZonalInfluenceFunctionMeasurer
from tesi_ao.mems_reconstructor import MemsZonalReconstructor
from arte.utils.zernike_generator import ZernikeGenerator
import numpy as np
import matplotlib.pyplot as plt


class Prove220531():

    def __init__(self, ifs_fname, mcl_fname):

        self._mcl = MemsCommandLinearization.load(mcl_fname)
        self.ppstroke, self.actuators_list, self.ifs = ZonalInfluenceFunctionMeasurer.load_ifs(
            ifs_fname)
        self.ifs_mask = self.ifs[0].mask
        self.cmask_obj = CircularMask(
            self.ifs[0].shape, maskRadius=131., maskCenter=(234., 309.))
        self.cmask = self.cmask_obj.mask()
        self._wyko, self._bmc = create_devices()

    def _prova_get_rec_per_spianare(self):
        # cmask_obj = CircularMask(
        #     self.ifs[0].shape, maskRadius=131., maskCenter=(234., 309.))
        # cmask = cmask_obj.mask()
        mzr = MemsZonalReconstructor(
            self.cmask, self.ppstroke, self.actuators_list, self.ifs)
        rec = mzr._rec
        return rec

    def one_step(self, rec):
        gain = 1
        wf = self._wavefront_on_cmask()
        self._dpos = np.zeros(140)
        self._dpos = -1 * \
            gain * np.dot(rec, wf.compressed())
        currentpos = self._mcl.c2p(self._bmc.get_shape())
        self._bmc.set_shape(self._mcl.p2c(currentpos + self._dpos))
        wf_spianato = self._wavefront_on_cmask()
        err_wf_sp = (wf_spianato.compressed()).std()
        print(err_wf_sp)
        plt.figure()
        plt.clf()
        plt.imshow(wf_spianato)
        plt.colorbar()

    def _wavefront_on_cmask(self):
        wf = self._wyko.wavefront(timeout_in_sec=10)
        wf = np.ma.array(wf.data, mask=self.cmask)
        return wf

    def deforma_mems(self):
        n_acts = len(self.actuators_list)
        pos = np.zeros(n_acts)
        pos = np.random.uniform(
            -500e-9, 500e-9, n_acts)
        self._bmc.set_shape(self._mcl.p2c(pos))

    def prova_genera_rec_con_cmask(self):
        '''
        creo una maschera circolare e costruisco la matrice di ricostruzione
        con quella maschera
        verifico che la cmask sia all interno della ifs_mask

        '''
        # pmb = PupilMaskBuilder(self.ifs_mask)
        # yc, xc = pmb.get_barycenter_of_false_pixels()
        # yc = yc - yc % 1.
        # xc = xc - xc % 1.
        # n_pixel_along_axis = pmb.get_number_of_false_pixels_along_pixel_axis(
        #     yc, xc)
        # radius_in_pixels = np.min(n_pixel_along_axis) * 0.5 - 2.
        # radius_in_pixels = radius_in_pixels - radius_in_pixels % 1.
        # self.cmask_obj = pmb.get_circular_mask(radius_in_pixels, (yc, xc))
        # print(self.cmask_obj)
        # self.cmask = self.cmask_obj.mask()
        self.cmask_obj = CircularMask.fromMaskedArray(self.ifs[0])
        print(self.cmask_obj)
        self.cmask = self.cmask_obj.mask()
        mzr = MemsZonalReconstructor(
            cmask=self.cmask, ifs_stroke=self.ppstroke, act_list=self.actuators_list, ifs=self.ifs)
        self.rec = mzr.get_reconstruction_matrix()
        self.im = mzr._im

    def prova_genera_zernike_in_cmask(self, j, aj):
        zg = ZernikeGenerator(self.cmask_obj)
        wfmode = np.zeros(self.cmask.shape)
        wfmode = np.ma.array(data=wfmode, mask=self.cmask)
        #j = 4
        #aj = 150.e-9
        z_mode = aj * zg.getZernike(j)
        wfmode = np.ma.array(data=z_mode.data, mask=self.cmask)
        plt.figure()
        plt.clf()
        plt.title('Generated Z%d' % j + ' aj = %g m' % aj, size=25)
        plt.imshow(wfmode)
        plt.colorbar()
        pos_cmd_from_wfmode = np.dot(self.rec, wfmode[self.cmask == False])
        print(pos_cmd_from_wfmode.shape)
        for idx in range(len(pos_cmd_from_wfmode)):
            max_stroke = np.max(
                self._mcl._deflection[self.actuators_list[idx]])
            min_stroke = np.min(
                self._mcl._deflection[self.actuators_list[idx]])
            if(pos_cmd_from_wfmode[idx] > max_stroke):
                pos_cmd_from_wfmode[idx] = max_stroke
                print('act%d reached max stroke' % self.actuators_list[idx])
            if(pos_cmd_from_wfmode[idx] < min_stroke):
                pos_cmd_from_wfmode[idx] = min_stroke
                print('act%d reached min stroke' % self.actuators_list[idx])
        wffitted = np.zeros(self.cmask.shape)
        wffitted[self.cmask == False] = np.dot(self.im, pos_cmd_from_wfmode)
        wffitted = np.ma.array(
            data=wffitted, mask=self.cmask)
        plt.figure()
        plt.clf()
        plt.title('Fitted Z%d' % j + ' aj = %g m' % wffitted.std(), size=25)
        plt.imshow(wffitted)
        plt.colorbar()
        plt.figure()
        plt.clf()

        plt.imshow(wfmode - wffitted)
        plt.colorbar()
        expected_fitting_error = (wfmode - wffitted).std()
        plt.title('Difference gen - exp Z%d' % j + ' fit err = %g m' %
                  expected_fitting_error, size=25)
        print('exp fit err: %g' % expected_fitting_error)


class Prove2205020():

    def __init__(self, mcl_name, cplm_name):
        #mcl_name = '/Users/lbusoni/Downloads/mcl_all_fixedpix.fits'
        #cplm_name = '/Users/lbusoni/Downloads/cplm_mcl_all_fixed-1/cplm_all_fixed.fits'
        self._wyko, self._bmc = create_devices()
        self._mcl = MemsCommandLinearization.load(mcl_name)
        cpla = CommandToPositionLinearizationAnalyzer(cplm_name)
        self._mg = ModeGenerator(cpla, self._mcl)

        wf = self._wyko.wavefront()
        self._mask = CircularMask(
            wf.shape, maskRadius=120, maskCenter=(235, 310))
        self._mg.THRESHOLD_RMS = 0.1
        self._mg.compute_reconstructor(self._mask)

    def spiana(self):
        for i in range(10):
            self.one_step()

    def one_step(self):
        gain = 1
        wf = self._wavefront_on_mask()
        self._dpos = np.zeros(140)
        self._dpos[self._mg._acts_in_pupil] = -1 * \
            gain * np.dot(self._mg._rec, wf.compressed())
        currentpos = self._mcl.c2p(self._bmc.get_shape())
        self._bmc.set_shape(self._mcl.p2c(currentpos + self._dpos))

    def _wavefront_on_mask(self):
        wf = self._wyko.wavefront()
        wf = np.ma.array(wf.data, mask=self._mask.mask())
        return wf

    def deforma(self):
        pos = np.zeros(140)
        pos[self._mg._acts_in_pupil] = np.random.uniform(
            -1000e-9, 1000e-9, self._mg._n_of_selected_acts)
        self._bmc.set_shape(self._mcl.p2c(pos))
