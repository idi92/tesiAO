import numpy as np
from tesi_ao.mems_command_linearization import MemsCommandLinearization
from tesi_ao.mems_zonal_influence_functions_measurer import ZonalInfluenceFunctionMeasurer
from tesi_ao.mems_reconstructor import MemsZonalReconstructor
from tesi_ao.main220316 import create_devices
from arte.types.mask import CircularMask
from astropy.io import fits
from arte.atmo import phase_screen_generator


class MemsFlatReshaper():

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
        self._rand_pos = 0
        self._psg = phase_screen_generator.PhaseScreenGenerator(
            240, 8, 40)
        self.set_reference_wavefront(None)

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

    def one_step(self, wavefront=None, gain=1):
        if wavefront is None:
            wavefront = self._wyko.wavefront(timeout_in_sec=10)
        if self.reference_wavefront() is not None:
            wavefront = wavefront - self.reference_wavefront()
        self._wf = self._wavefront_on_cmask(wavefront)
        self._dpos = np.zeros(self.number_of_actuators)
        self._dpos[self._mzr.selected_actuators] = -1 * \
            gain * np.dot(self.rec, self._wf.compressed())
        currentpos = self._mcl.c2p(self._bmc.get_shape())
        self._bmc.set_shape(self._mcl.p2c(currentpos + self._dpos))
        # self.show_wf_and_print_residual()

    def show_wf_and_print_residual(self):
        import matplotlib.pylab as plt
        wf_spianato = self._wavefront_on_cmask(
            self._wyko.wavefront(timeout_in_sec=10))
        err_wf_sp = (wf_spianato.compressed()).std()
        print(err_wf_sp)
        plt.figure()
        plt.clf()
        plt.imshow(wf_spianato)
        plt.colorbar()

    @property
    def number_of_actuators(self):
        return self._bmc.get_number_of_actuators()

    @property
    def number_of_selected_actuators(self):
        return self._mzr.number_of_selected_actuators

    @property
    def selected_actuators(self):
        return self._mzr.selected_actuators

    def _check_wavefront_mask_is_in_cmask(self, wf_mask):
        intersection_mask = np.ma.mask_or(wf_mask, self.cmask)
        assert (intersection_mask == self.cmask).all(
        ) == True, "Wavefront mask is not valid!\nShould be fully inscribed in the reconstructor mask!"

    def _wavefront_on_cmask(self, wf):
        # wf = self._wyko.wavefront(timeout_in_sec=10)
        # TODO: add check that wf.mask is included in self.cmask
        # otherwise raise exception
        self._check_wavefront_mask_is_in_cmask(wf.mask)
        wf = np.ma.array(wf.data, mask=self.cmask)
        return wf

    def apply_random_distortion(self, stroke=500e-9):
        '''
        applies random actuation between[-stroke,stroke] to all actuators
        '''

        pos = np.random.uniform(
            -stroke, stroke, self.number_of_actuators)
        self._bmc.set_shape(self._mcl.p2c(pos))
        self._rand_pos = stroke

    def _generate_wavefront_phase_screen(self, r0InMAt500nm):
        self._psg._seed += 1
        self._psg.generate_normalized_phase_screens(1)
        self._psg.rescale_to(r0InMAt500nm)
        ps = self._psg.get_in_meters()
        wf = np.zeros((486, 640))
        wf[231 - 120:231 + 120, 306 - 120:306 + 120] = ps
        return np.ma.array(data=wf, mask=self.cmask)

    def apply_phase_screen_distortion(self, r0=0.1):
        phase_screen_wf = self._generate_wavefront_phase_screen(r0)
        self.set_reference_wavefront(phase_screen_wf)

    def reference_wavefront(self):
        return self._ref_wf

    def set_reference_wavefront(self, ref_wf):
        self._ref_wf = ref_wf

    def do_some_statistic(self, distorted_wf, n_of_flattening_steps=10):
        n_meas = n_of_flattening_steps + 1
        #steps = np.arange(n_meas)
        frame_shape = distorted_wf.mask.shape
        self.wf_meas = np.ma.zeros((n_meas, frame_shape[0], frame_shape[1]))
        self.wf_meas[0] = distorted_wf
        self._cmd_vector = np.zeros((n_meas, self.number_of_actuators))
        self._cmd_vector[0] = self._bmc.get_shape()
        for i in range(1, n_meas):
            self.one_step()
            self._cmd_vector[i] = self._bmc.get_shape()
            self.wf_meas[i] = self._wavefront_on_cmask(
                self._wyko.wavefront(timeout_in_sec=10))

    def save_stats(self, fname):
        hdr = fits.Header()
        hdr['STROKE'] = self._rand_pos
        hdr['THRES'] = self._mzr.THRESHOLD_RMS
        # hdr['CENTER'] = self.cmask_obj.center()
        fits.writeto(fname, self.wf_meas.data, hdr)
        fits.append(fname, self.wf_meas.mask.astype(int))
        fits.append(fname, self._cmd_vector)

    @staticmethod
    def load_stat(fname):
        header = fits.getheader(fname)
        hduList = fits.open(fname)
        wfmeas_data = hduList[0].data
        wfmeas_mask = hduList[1].data.astype(bool)
        wfs_meas = np.ma.masked_array(data=wfmeas_data, mask=wfmeas_mask)
        cmd_vector = hduList[2].data
        rand_stroke = header['STROKE']
        thres = header['THRES']
        # cmask_center = header['CENTER']
        return wfs_meas, cmd_vector, rand_stroke, thres

    def load_acquired_measures_4plot(self):
        cpath1 = 'prova/misure_con_tappo/misure_spianamento/thres'
        thres_label = ['0000', '0008', '0015', '0025', '0050']
        n_thres = len(thres_label)
        start_wf_label = ['wfflat', 'wfrand500nm', 'wfrand1000nm']
        n_start_wf = len(start_wf_label)
        n_times = 10
        n_points = 11
        self.n_points = n_points
        self.n_flatten = 10
        self.flatten_data = np.zeros((n_thres, n_start_wf, n_times, n_points))
        self.thres_list = np.zeros(n_thres)
        self.pos_list = np.zeros(n_start_wf)
        for idx_thres, folder in enumerate(thres_label):
            for idx_wf, wfstart in enumerate(start_wf_label):
                for t in range(n_times):
                    fname = cpath1 + folder + '/mfr_stat' + wfstart + \
                        '_thres' + folder + '_time%d' % t + '.fits'
                    wfs_meas, cmd_vector, rand_stroke, thres = MemsFlatReshaper.load_stat(
                        fname)
                    self.flatten_data[idx_thres, idx_wf,
                                      t] = wfs_meas.std(axis=(1, 2))
                self.pos_list[idx_wf] = rand_stroke
            self.thres_list[idx_thres] = thres

    def save_acquired_measures_4plot(self, fname):
        hdr = fits.Header()
        hdr['N_SP'] = self.n_flatten
        hdr['N_PT'] = self.n_points
        fits.writeto(fname, self.flatten_data, hdr)
        fits.append(fname, self.pos_list)
        fits.append(fname, self.thres_list)

    @staticmethod
    def load_flatten_data(fname):
        header = fits.getheader(fname)
        hduList = fits.open(fname)
        flatten_data = hduList[0].data
        pos_list = hduList[1].data
        thres_list = hduList[2].data
        n_points = header['N_PT']
        n_flatten = header['N_SP']
        # cmask_center = header['CENTER']
        return flatten_data, pos_list, thres_list, n_points, n_flatten
