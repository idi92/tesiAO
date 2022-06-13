import numpy as np
from astropy.io import fits
from tesi_ao.mems_command_linearization import MemsCommandLinearization
from tesi_ao.mems_zonal_influence_functions_measurer import ZonalInfluenceFunctionMeasurer
from tesi_ao.main220316 import create_devices
from arte.types.mask import CircularMask
from tesi_ao.mems_reconstructor import MemsZonalReconstructor
from tesi_ao.zernike_modal_reconstructor import ZernikeToZonal
from arte.utils.zernike_generator import ZernikeGenerator


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

    def create_reconstructor(self, set_thresh=0.25):
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
        # amplitude_span = 1e-9 * \
        #     np.array([10, 25, 50, 100, 300, 500, 800, 1000, 1250])
        self._j = j
        amplitude_span = 1e-9 * np.array([100, 500, 1000])
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
            wf_flat = self.get_wavefront()
            self.apply_zernike(j, aj)
            self._check_clipped_acts[idx] = self._mcl.clipping_vector
            self.wfs[idx] = self.get_wavefront() - wf_flat
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

    def __init__(self, fname):
        self.wfs,  self.z_mode, self.exp_amp_vector, self.pos_vector, self._clip_vector, \
            self.selected_act, self.mode_index,\
            self._thres_rms = MemsModeMeasurer.load(fname)

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

    def show_mode_linearity(self):
        return 0
