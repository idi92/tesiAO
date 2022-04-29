import numpy as np
import matplotlib.pyplot as plt


class MemsfittingError():

    def __init__(self, r0, diameter, firstNmodes):
        self._dr0_ratio = diameter / r0
        self._ratio53 = (diameter / r0)**(5. / 3.)
        self._num_of_modes = firstNmodes - 1
        # avoid Z1
        self._modes_list = np.arange(2, firstNmodes + 1)
        self._firstNmodes = firstNmodes
        self._wavelength = 632.8e-9  # meters
        self._create_zernike_variance()

    def estimate_expected_fitting_error(self, mg, mcl, pupil_mask_obj):
        self._expected_fitting_error = np.zeros(self._num_of_modes)
        # posso passare direttamente mg col
        # THRESHOLD_RATIO che voglio
        # mg.THRESHOLD_RATIO = 0.5
        mg.compute_reconstructor(mask_obj=pupil_mask_obj)
        for idx, j in enumerate(self._modes_list):
            # coef_a [rad] --> aj [nm rms]
            aj = self._coef_a[idx] * 0.5 * self._wavelength / np.pi
            print('Creating Z%d' % int(j) + ' with aj=%g [m]' % aj)
            mg.generate_zernike_mode(int(j), aj)
            mg.build_fitted_wavefront()
            self._expected_fitting_error[idx] = (
                mg._wffitted - mg._wfmode).std()
            self._acts_in_pupil = mg._acts_in_pupil

    def _plot_fitting_error(self):
        total_amplitude = self._coef_a.sum() * 0.5 * self._wavelength / np.pi
        total_amplitude = np.sqrt(self._get_delta_from_noll(
            1)) * 0.5 * self._wavelength / np.pi
        num_of_act = len(self._acts_in_pupil)
        plt.figure()
        plt.clf()
        plt.plot(self._modes_list, self._expected_fitting_error /
                 total_amplitude, 'o-', label='#Nact = %d' % num_of_act)
        plt.xlabel('Zernike index j', size=25)
        plt.ylabel('Normalized expected fitting error', size=25)
        total_amplitude = total_amplitude / 1.e-9,
        plt.title('D/r0=%g ' % self._dr0_ratio + 'A = %g nm' %
                  total_amplitude, size=25)
        plt.legend(loc='best')

    def _plot_cumulative_variace(self):
        # const = ( 0.5 * self._wavelength / np.pi)**2
        residual_variance1 = self._get_delta_from_noll(J=1)
        plt.figure()
        plt.clf()
        plt.title('First %d' % self._firstNmodes + ' Zernike generated')
        for idx, j in enumerate(self._modes_list):
            cumulative_variance = self._var_a[0:idx].sum() / residual_variance1
            plt.plot(j, cumulative_variance, 'o-')
        plt.xlabel('Zernike index j', size=25)
        plt.ylabel('Normalized cumulative expected variance', size=25)
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
