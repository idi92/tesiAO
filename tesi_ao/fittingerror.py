import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


class MemsfittingError():

    WAVELENGTH = 632.8e-9
    const = 0.5 * WAVELENGTH / np.pi  # from rad to nm

    def __init__(self, r0, diameter, firstNmodes, j_start=None):
        if j_start is None:
            # avoid Z1
            j_start = 2
        self._dr0_ratio = diameter / r0
        self._ratio53 = (diameter / r0)**(5. / 3.)
        self._modes_list = np.arange(j_start, firstNmodes + 1)
        self._num_of_modes = len(self._modes_list)
        self._firstNmodes = firstNmodes
        self._wavelength = 632.8e-9  # meters
        self._create_zernike_variance()

    def compute_expected_fitting_error(self, mg, mcl, pupil_mask_obj):
        self._expected_fitting_error = np.zeros(self._num_of_modes)
        # posso passare direttamente mg col
        # THRESHOLD_RATIO che voglio
        # mg.THRESHOLD_RATIO = 0.5
        mg.compute_reconstructor(mask_obj=pupil_mask_obj)
        for idx, j in enumerate(self._modes_list):
            # coef_a [rad] --> aj [nm rms]
            aj = self._coef_a[idx] * self.const
            print('Creating Z%d' % int(j) + ' with aj=%g [m]' % aj)
            mg.generate_zernike_mode(int(j), aj)
            mg.build_fitted_wavefront()
            self._expected_fitting_error[idx] = (
                mg._wffitted - mg._wfmode).std()
            self._acts_in_pupil = mg._acts_in_pupil
        self._expected_fitting_variance = self._expected_fitting_error**2

    def _show_normalized_fitting_error_pattern(self):
        '''
        plots the fitting errors normalized to the relative
        mode amplitude aj, as a function of j index
        '''
        num_of_act = len(self._acts_in_pupil)
        plt.figure()
        plt.clf()
        aj = self._coef_a * self.const
        plt.plot(self._modes_list, self._expected_fitting_error /
                 aj, 'o-', label='#Nact = %d' % num_of_act)
        plt.xlabel('Zernike index j', size=25)
        plt.ylabel(r'$\sigma_{fitting_j}/a_j$', size=25)
        plt.title(r'$D/r_0=%g $' % self._dr0_ratio + '\t' + r'$\lambda = %g m$' %
                  self.WAVELENGTH, size=25)
        plt.legend(loc='best')
        plt.grid()

    def _compute_cumulative_rms(self):
        const2 = self.const * self.const
        #residual_variance1 = const2 * self._get_delta_from_noll(J=1)

        self._cumulative_rms = np.zeros(self._num_of_modes)
        fitted_variance = self._expected_fitting_error**2
        for idx, j in enumerate(self._modes_list):
            quad_sum = self._get_delta_from_noll(
                J=int(j)) * const2 + fitted_variance[0:idx + 1].sum()
            self._cumulative_rms[idx] = np.sqrt(quad_sum)

    def _show_normalized_cumulative(self):
        const2 = self.const * self.const
        residual_variance1 = const2 * self._get_delta_from_noll(J=1)
        sigma1 = np.sqrt(residual_variance1)
        plt.figure()
        plt.clf()
        plt.title('First %d' % self._firstNmodes + ' Zernike generated' + 'D/r0=%g' %
                  self._dr0_ratio + ' lambda=%g m' % self.WAVELENGTH)

        plt.loglog(self._modes_list, self._cumulative_rms / sigma1, 'bo-')
        plt.xlabel('Zernike index j', size=25)
        plt.ylabel('Normalized cumulative rms', size=25)
        plt.grid()

    def _create_zernike_kolmogoroff_residual_errors(self):
        '''
        deltaJ from Noll1976
        '''
        self._residual_error = np.zeros(self._firstNmodesN)
        for idx in range(self._firstNmodes):
            j = idx + 1
            self._residual_error[idx] = self._get_delta_from_noll(j)

    def _create_zernike_variance(self):
        self._var_a = np.zeros(self._num_of_modes)

        for idx, j in enumerate(self._modes_list):
            self._var_a[idx] = self._get_delta_from_noll(
                int(j - 1)) - self._get_delta_from_noll(int(j))
        self._coef_a = np.sqrt(self._var_a)

    def _get_delta_from_noll(self, J):
        '''
        returns Zernike_kolmogoroff residual error deltaJ
        [rad2]
        (Noll1976)
        '''
        assert J > 0, "Avoid piston divergence! Must be J>=1!"
        if(J == 1):
            deltaJ = 1.0299 * self._ratio53
        if(J == 2):
            deltaJ = 0.582 * self._ratio53
        if(J == 3):
            deltaJ = 0.134 * self._ratio53
        if(J == 4):
            deltaJ = 0.111 * self._ratio53
        if(J == 5):
            deltaJ = 0.0880 * self._ratio53
        if(J == 6):
            deltaJ = 0.0648 * self._ratio53
        if(J == 7):
            deltaJ = 0.0587 * self._ratio53
        if(J == 8):
            deltaJ = 0.0525 * self._ratio53
        if(J == 9):
            deltaJ = 0.0463 * self._ratio53
        if(J == 10):
            deltaJ = 0.0401 * self._ratio53
        if(J == 11):
            deltaJ = 0.0377 * self._ratio53
        if(J == 12):
            deltaJ = 0.0352 * self._ratio53
        if(J == 13):
            deltaJ = 0.0328 * self._ratio53
        if(J == 14):
            deltaJ = 0.0304 * self._ratio53
        if(J == 15):
            deltaJ = 0.0279 * self._ratio53
        if(J == 16):
            deltaJ = 0.0267 * self._ratio53
        if(J == 17):
            deltaJ = 0.0255 * self._ratio53
        if(J == 18):
            deltaJ = 0.0243 * self._ratio53
        if(J == 19):
            deltaJ = 0.0232 * self._ratio53
        if(J == 20):
            deltaJ = 0.0220 * self._ratio53
        if(J == 21):
            deltaJ = 0.0208 * self._ratio53
        if (J > 21):
            deltaJ = 0.2944 * J**(-np.sqrt(3.) / 2.) * self._ratio53
        return deltaJ

    def _plot_normalized_variance(self):
        plt.figure()
        plt.clf()
        plt.loglog(self._modes_list, self._var_a / self._ratio53, 'o')
        plt.xlabel('Zernike index j', size=25)
        plt.ylabel('normalized variance', size=25)
        plt.grid()


class MemsAmplitudeLinearityEstimator():
    AMPLITUDE_SPAN = np.array([-2000., -1500., -1000., -800., -400., -200., -100., -
                               50., -40., -20., 20., 40., 50., 100., 200., 400., 800., 1000., 1500., 2000.]) * 1.e-9
    fpath = 'prova/misure_ampiezze/male_'
    ffmt = '.fits'

    def __init__(self, mcl, mg, mm, pupil_mask_obj):
        self._calibration = mcl
        self._mode_generator = mg
        self._mode_measurer = mm
        self._pupil_mask = pupil_mask_obj

    def _compute_measured_amplitude(self, jmode, expected_amp):

        self._mode_generator.generate_zernike_mode(int(jmode), expected_amp)
        self._mode_generator.build_fitted_wavefront()
        # fitted_amplitude = (self._mode_generator._wffitted).std()
        # expected_fitting_error = (
        # self._mode_generator._wffitted - self._mode_generator._wfmode).std()
        self._mode_measurer.execute_measure(
            self._calibration, self._mode_generator)
        measured_amplitude = (self._mode_measurer._wfmeas).std()
        measured_fitting_error = (
            self._mode_measurer._wfmeas - self._mode_generator._wfmode).std()
        return measured_amplitude, measured_fitting_error

    def _compute_expected_fitting_amplitude(self, jmode, expected_amp):
        self._mode_generator.generate_zernike_mode(int(jmode), expected_amp)
        self._mode_generator.build_fitted_wavefront()
        fitted_amplitude = (self._mode_generator._wffitted).std()
        expected_fitting_error = (
            self._mode_generator._wffitted - self._mode_generator._wfmode).std()
        return fitted_amplitude, expected_fitting_error

    def execute_amplitude_linearity_measures(self, jmode):
        # magari posso farlo da terminale
        # o un for al variare del THRESHOLD_RMS
        # self._mode_generator.THRESHOLD_RMS = threshold
        self._mode_generator.compute_reconstructor(self._pupil_mask)
        self._measured_amplitude = np.zeros(len(self.AMPLITUDE_SPAN))
        self._measured_fitting_error = np.zeros_like(self._measured_amplitude)
        for idx, aj in enumerate(self.AMPLITUDE_SPAN):
            self._measured_amplitude[idx], self._measured_fitting_error[idx] = self._compute_measured_amplitude(
                jmode, aj)

    def _show_measured_vs_generated_amplitude(self, jmode):
        plt.figure()
        plt.clf()
        plt.title(r'$Mode:\t Z_{%d}$' % jmode)
        x = self.AMPLITUDE_SPAN / 1.e-9
        y = self._measured_amplitude
        yerr = self._measured_fitting_error
        plt.errorbar(x, y, yerr)
        plt.xlabel(r'$A_{generated}$\t[nm]', size=25)
        plt.ylabel(r'$A_{measured}$\t[nm]', size=25)

    def save_results(self, jmode):
        file_name = self.fpath + 'z%d' % jmode + self.ffmt
        hdr = fits.Header()
        hdr['J_MODE'] = jmode
        fits.writeto(file_name, self.AMPLITUDE_SPAN, hdr)
        fits.append(file_name, self._measured_amplitude)
        fits.append(file_name, self._measured_fitting_error)

    def load(self, jmode):
        fname = self.fpath + 'z%d' % jmode + self.ffmt
        header = fits.getheader(fname)
        hduList = fits.open(fname)
        self.AMPLITUDE_SPAN = hduList[0]
        self._measured_amplitude = hduList[1]
        self._measured_fitting_error = hduList[2]
