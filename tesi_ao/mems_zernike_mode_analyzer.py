import numpy as np
from astropy.io import fits
from tesi_ao.mems_command_linearization import MemsCommandLinearization
from tesi_ao.mems_zonal_influence_functions_measurer import ZonalInfluenceFunctionMeasurer
from tesi_ao.main220316 import create_devices
from arte.types.mask import CircularMask
from tesi_ao.mems_reconstructor import MemsZonalReconstructor
from tesi_ao.zernike_modal_reconstructor import ZernikeToZonal
from arte.utils.zernike_generator import ZernikeGenerator
from arte.utils import modal_decomposer
from arte.types.wavefront import Wavefront


class MemsModeMeasurer():

    def __init__(self, ifs_fname=None, mcl_fname=None):
        if ifs_fname is None:
            ifs_fname = 'prova/misure_con_tappo/zonal_ifs/zifs_pushpull500nmtime0_mcl_def.fits'
        if mcl_fname is None:
            mcl_fname = 'prova/misure_ripetute/mcl_all_def.fits'

        self._mcl = MemsCommandLinearization.load(mcl_fname)
        self.ppstroke, self.list_all_acts, self.ifs = ZonalInfluenceFunctionMeasurer.load_ifs(
            ifs_fname)
        self.ifs_mask = self.ifs[0].mask
        self._wyko, self._bmc = create_devices()

    def create_mask(self, radius=120, center=(231, 306)):
        '''
        Creates a circular mask with given radius and center
        default values: 
        radius=120 
        center=(231,306) center of central actuators
        '''
        self.cmask_obj = CircularMask(
            self.ifs[0].shape, maskRadius=radius, maskCenter=center)
        self.cmask = self.cmask_obj.mask()

    def create_mask_from_ifs(self):
        '''
        A CircularMask with radius 5% smaller than the one of the Influence
        Functions
        '''
        mask = CircularMask.fromMaskedArray(self.ifs[0])
        self.cmask_obj = CircularMask(mask.shape(),
                                      mask.radius() * 0.95,
                                      mask.center())
        print(self.cmask_obj)
        self.cmask = self.cmask_obj.mask()

    def create_reconstructor(self, set_thresh=0.15):
        '''
        costruisco la matrice di ricostruzione
        con la maschera specificata precedentemente
        (MemsModalReconstructor verifica che cmask sia inclusa in ifs_mask)

        '''
        self._mzr = MemsZonalReconstructor(
            cmask=self.cmask, ifs_stroke=self.ppstroke, ifs=self.ifs)
        self._mzr.THRESHOLD_RMS = set_thresh
        self.rec = self._mzr.reconstruction_matrix
        self.im = self._mzr.interaction_matrix

    def create_zernike2zonal(self):
        self.zern2zonal = ZernikeToZonal(self._mzr, self.cmask_obj)
        self.num_of_mems_modes = self.zern2zonal._n_zern_modes

    def apply_zernike(self, j, aj):
        modal_ampl = np.zeros(self.num_of_mems_modes)
        modal_ampl[j - 2] = aj
        self.pos = self.zern2zonal.convert_modal_to_zonal(modal_ampl)
        self._bmc.set_shape(self._mcl.p2c(self.pos))  # + self.cmd_flat)

        #self.wf = self.get_wavefront()

    def execute_amplitude_scan_for_zernike(self, j):
        amplitude_span = 1e-9 * \
            np.array([5, 10, 25, 50, 100, 300, 500,
                      650, 800, 1000, 1250, 1500])
        self._j = j
        #amplitude_span = 1e-9 * np.array([100, 500, 1000])
        self._amplitude_vector = np.append(-amplitude_span, amplitude_span)
        self._amplitude_vector.sort()
        num_of_meas = len(self._amplitude_vector)
        frame_shape = self.cmask.shape
        self.wfs = np.ma.zeros((num_of_meas, frame_shape[0], frame_shape[1]))
        cmd_flat = np.zeros(self.number_of_actuators)
        self._pos_vector = np.zeros((num_of_meas, self.number_of_actuators))
        self._check_clipped_acts = np.zeros(
            (num_of_meas, self.number_of_actuators))
        print('Z%d applied:' % j)

        for idx, aj in enumerate(self._amplitude_vector):
            print('aj = %g' % aj)
            self._bmc.set_shape(cmd_flat)
            try:
                wf_flat = self.get_wavefront()
            except AssertionError:
                wf_flat = self.get_wavefront()

            self.apply_zernike(j, aj)
            self._check_clipped_acts[idx] = self._mcl.clipping_vector
            try:
                actual_wf = self.get_wavefront()
            except AssertionError:
                actual_wf = np.ma.array(data=np.zeros(
                    frame_shape), mask=np.ones(frame_shape, dtype=np.bool))
            self.wfs[idx] = actual_wf - wf_flat
            self._pos_vector[idx] = self.pos
            self._bmc.set_shape(cmd_flat)

    def show_plot(self):
        return 0

    @property
    def number_of_actuators(self):
        return self._bmc.get_number_of_actuators()

    @property
    def selected_actuators(self):
        return self._mzr.selected_actuators

    @property
    def number_of_selected_actuators(self):
        return self._mzr.number_of_selected_actuators

    @property
    def get_actuators_visibility_threshold(self):
        return self._mzr.THRESHOLD_RMS

    def _check_wavefront_mask_is_in_cmask(self, wf_mask):
        intersection_mask = np.ma.mask_or(wf_mask, self.cmask)
        assert (intersection_mask == self.cmask).all(
        ) == True, "Wavefront mask is not valid!\nShould be fully inscribed in the reconstructor mask!"

    def _wavefront_on_cmask(self):
        wf = self._wyko.wavefront(timeout_in_sec=10)
        # TODO: add check that wf.mask is included in self.cmask
        # otherwise raise exception
        self._check_wavefront_mask_is_in_cmask(wf.mask)
        wf = np.ma.array(wf.data, mask=self.cmask)
        return wf

    def get_wavefront(self):
        return self._wavefront_on_cmask()

    def get_expected_zernike(self, j=None):
        if j is None:
            j = self._j
        zg = ZernikeGenerator(self.cmask_obj)
        zmode = zg.getZernike(self._j)
        return zmode

    def save_results(self, fname):

        hdr = fits.Header()
        hdr['JMODE'] = self._j
        hdr['THRES'] = self.get_actuators_visibility_threshold

        fits.writeto(fname, self.wfs.data, hdr)
        fits.append(fname, self.wfs.mask.astype(int))

        zmode = self.get_expected_zernike(j=self._j)
        fits.append(fname, zmode.data)
        fits.append(fname, zmode.mask.astype(int))

        fits.append(fname, self._amplitude_vector)
        fits.append(fname, self._pos_vector)
        fits.append(fname, self._check_clipped_acts)
        fits.append(fname, self.selected_actuators)

    @staticmethod
    def load(fname):
        header = fits.getheader(fname)
        hduList = fits.open(fname)
        wfs_data = hduList[0].data
        wfs_mask = hduList[1].data.astype(bool)
        wfs = np.ma.masked_array(data=wfs_data, mask=wfs_mask)
        zmode_data = hduList[2].data
        zmode_mask = hduList[3].data.astype(bool)
        z_mode = np.ma.masked_array(data=zmode_data, mask=zmode_mask)
        amplitude_vector = hduList[4].data
        pos_vector = hduList[5].data
        clip_vector = hduList[6].data
        selected_act = hduList[7].data
        mode_index = header['JMODE']
        thres_rms = header['THRES']
        return wfs, z_mode, amplitude_vector, pos_vector, clip_vector, selected_act, mode_index, thres_rms


class MemsAmplitudeLinearityAnalizer():
    NUM_DECOMPOSITION_MODES = 99

    def __init__(self, fname):
        self.wfs,  self.z_mode, self.exp_amp_vector, self.pos_vector, self._clip_vector, \
            self.selected_acts, self.mode_index,\
            self._thres_rms = MemsModeMeasurer.load(fname)
        mmm = MemsModeMeasurer(ifs_fname=None, mcl_fname=None)
        mmm.create_mask(radius=120, center=(231, 306))
        self.mask = mmm.cmask_obj
        self.md = modal_decomposer.ModalDecomposer(
            self.NUM_DECOMPOSITION_MODES)

    def get_measured_amplitudes(self):
        self.meas_amp_vector = self.wfs.std(axis=(1, 2))
        return self.meas_amp_vector

    def get_expected_amplitudes(self):
        return self.exp_amp_vector

    def get_measured_fitting_error(self):
        n_meas = self.wfs.shape[0]
        fitting_error = np.zeros(n_meas)
        for i in range(n_meas):
            fitting_error[i] = (
                self.wfs[i] - self.exp_amp_vector[i] * self.z_mode).std()
        return fitting_error

    def show_fitting_map(self, amp_index):

        aj = self.get_expected_amplitudes()[amp_index]
        expected_wf = aj * self.z_mode / 1e-9
        measured_wf = self.wfs[amp_index] / 1e-9
        fitting_error_map = measured_wf - expected_wf
        import matplotlib.pyplot as plt
        plt.figure()
        plt.clf()
        plt.imshow(measured_wf)
        plt.colorbar()
        plt.figure()
        plt.clf()
        plt.imshow(fitting_error_map)
        plt.colorbar(label='[nm]')
        plt.figure()
        plt.clf()
        plt.plot(fitting_error_map[231, :])
        plt.plot(fitting_error_map[:, 306])
        plt.plot(expected_wf[231, :])
        plt.plot(expected_wf[:, 306])
        plt.plot(measured_wf[231, :])
        plt.plot(measured_wf[:, 306])
        plt.grid()

    def show_mode_linearity(self):
        import matplotlib.pyplot as plt

        y = self.get_measured_amplitudes()
        sigma = self.get_measured_fitting_error()
        x = self.get_expected_amplitudes()
        sign = np.ones_like(y)
        half = len(sign) * 0.5
        sign[:int(half)] = -1
        y = sign * y
        plt.figure()
        plt.clf()
        plt.title('$Z_{%d}$' % self.mode_index + ' $thres = %g$' %
                  self._thres_rms + ' (%d acts)' % len(self.selected_acts), size=25)
        plt.xlabel('Expected Amplitude [nm]', size=20)
        plt.ylabel('Measured Amplitude [nm]', size=20)

        plt.plot(x / 1e-9, y / 1e-9, '-bo', label='Z%d' % self.mode_index)
        plt.errorbar(x / 1e-9, y / 1e-9, sigma / 1e-9, ls=None)

        idx_clip_modes = self.get_clipped_modes_index_list()
        plt.plot(x[idx_clip_modes] / 1e-9, y[idx_clip_modes] /
                 1e-9, 'ro', label='clipped')
        plt.grid()
        plt.legend(loc='best')

    def decompose_wavefront_on_zernike(self, wf):
        import matplotlib.pyplot as plt
        try:
            zc = self.md.measureZernikeCoefficientsFromWavefront(
                Wavefront(wf), self.mask)
        except ValueError:
            return np.zeros(self.NUM_DECOMPOSITION_MODES)
        #plt.plot(zc.zernikeIndexes(), zc.toNumpyArray())
        return zc.toNumpyArray()

    # def show_fitting_error_pattern(self):
    #     import matplotlib.pyplot as plt
    #     y = self.get_measured_fitting_error()
    #     x = self.get_expected_amplitudes()
    #     #x = self.get_measured_amplitudes()
    #     sign = np.ones_like(y)
    #     half = len(sign) * 0.5
    #     sign[:int(half)] = -1
    #     #y = sign * y
    #     #x = sign * x
    #     plt.figure()
    #     plt.clf()
    #     plt.title('$Z_{%d}$' % self.mode_index + ' $thres = %g$' %
    #               self._thres_rms + ' (%d acts)' % len(self.selected_acts), size=25)
    #     plt.xlabel('Measured Amplitude [nm]', size=20)
    #     plt.ylabel('Fitting error rms [nm]', size=20)
    #     plt.plot(x / 1e-9, y / 1e-9, '-bo', label='Z%d' % self.mode_index)
    #     idx_clip_modes = self.get_clipped_modes_index_list()
    #     plt.plot(x[idx_clip_modes] / 1e-9, y[idx_clip_modes] /
    #              1e-9, 'ro', label='clipped')
    #     plt.grid()
    #     plt.legend(loc='best')

    def get_clipped_modes_index_list(self):
        n_meas = self.wfs.shape[0]
        n_all_acts = self._clip_vector.shape[-1]
        unclipped_vector = np.zeros(n_all_acts)
        clip_mode_index = []
        for idx in range(n_meas):
            if((self._clip_vector[idx] == unclipped_vector).all() == False):
                clip_mode_index.append(idx)
        return clip_mode_index
