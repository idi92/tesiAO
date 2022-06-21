import numpy as np
from astropy.io import fits
from tesi_ao.main220316 import create_devices
from tesi_ao.mems_command_to_position_linearization_measurer import CommandToPositionLinearizationMeasurer
from tesi_ao.mems_command_to_position_linearization_analyzer import CommandToPositionLinearizationAnalyzer
from tesi_ao.mems_command_linearization import MemsCommandLinearization
from scipy.optimize import curve_fit


class InterpolationErrorAnalyzer():

    fpath1 = 'prova/mems_interpolation_error/iea_cplm_act'
    fpath2 = '_nscans'
    fversion = '_220616'
    fpath_mcl = 'prova/mems_interpolation_error/iea_mcl_act'
    ffmt = '.fits'

    def __init__(self, act, scan_list=None, lin_test_points=10):

        if scan_list is None:
            scan_list = np.array([10, 20, 30, 40, 50, 60, 100])
        self.scan_list = scan_list
        self.act = act
        self._wyko, self._bmc = create_devices()
        self.n_points = lin_test_points

    def execute_multiple_scans(self):
        cmd = np.zeros(self.number_of_actuators)
        for n_of_scans in self.scan_list:
            cplm = CommandToPositionLinearizationMeasurer(
                self._wyko, self._bmc)
            cplm.NUMBER_STEPS_VOLTAGE_SCAN = n_of_scans
            cplm.execute_command_scan([self.act])
            fname = self.fpath1 + '%d' % self.act + self.fpath2 + \
                '%d' % n_of_scans + self.fversion + self.ffmt
            cplm.save_results(fname)
            self._bmc.set_shape(cmd)

    def _get_cpla_from_file(self, fname):
        cpla = CommandToPositionLinearizationAnalyzer(fname)
        return cpla

    def _get_mcl_from_file(self, fname):
        cpla = self._get_cpla_from_file(fname)
        mcl = cpla.compute_linearization()
        return mcl

    def compute_multiple_scans_interpolation(self):
        for n_of_scans in self.scan_list:
            fname = self.fpath1 + '%d' % self.act + self.fpath2 + \
                '%d' % n_of_scans + self.fversion + self.ffmt
            mcl = self._get_mcl_from_file(fname)
            mcl_fname = self.fpath_mcl + '%d' % self.act + self.fpath2 + \
                '%d' % n_of_scans + self.fversion + self.ffmt
            mcl.save(mcl_fname)

    def load_multiple_scans_interpolation(self):
        mcl_list = []
        for n_of_scans in self.scan_list:
            mcl_fname = self.fpath_mcl + '%d' % self.act + self.fpath2 + \
                '%d' % n_of_scans + self.fversion + self.ffmt
            mcl = MemsCommandLinearization.load(mcl_fname)
            mcl_list.append(mcl)
        return mcl_list

    def _plot_interpolation_function(self, mcl):
        import matplotlib.pyplot as plt

        plt.plot(mcl._cmd_vector[0], mcl._deflection[0] / 1e-9, '.',
                 label='%d scans' % mcl._cmd_vector.shape[1])

        plt.plot(mcl._calibrated_cmd[0], mcl._calibrated_position[0] / 1e-9,
                 '-', color=plt.gca().lines[-1].get_color())

    def show_all_interpolation_functions(self, mcl_list):
        '''
        Plots all interpolated functions obtained by varying scan sampling,
        as a function of actuator's deflections.
        '''
        import matplotlib.pyplot as plt
        plt.figure()
        plt.clf()
        plt.ion()
        plt.title('act#%d: interpolation functions for several scans' %
                  self.act, size=15)
        for mcl in mcl_list:
            self._plot_interpolation_function(mcl)

        plt.xlabel('Commands [au]', size=15)
        plt.ylabel('Deflection [nm]', size=15)
        plt.grid()
        plt.legend(loc='best')

    def show_interpolation_difference(self, mcl_list):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.clf()
        plt.ion()
        plt.title('act#%d:' % self.act +
                  'cubic spline interpolation error w-r-t %dscans' % max(self.scan_list), size=15)
        mcl_ref = mcl_list[-1]
        for idx, scans in enumerate(self.scan_list):
            mcl = mcl_list[idx]
            dpos = mcl._calibrated_position[0] - \
                mcl_ref._calibrated_position[0]
            plt.plot(mcl._calibrated_cmd[0], dpos /
                     1e-9, '-', label='%d scans' % scans)
            print(dpos.std())

        plt.legend(loc='best')
        plt.grid()
        plt.ylabel('Deflection Difference [nm]', size=15)
        plt.xlabel('Commands [au]', size=15)

    def _get_common_position_range(self):
        mcl_list = self.load_multiple_scans_interpolation()
        min_container = []
        max_container = []
        for mcl in mcl_list:
            min_container.append(min(mcl._calibrated_position[0]))
            max_container.append(max(mcl._calibrated_position[0]))
        min_pos = max(min_container)
        max_pos = min(max_container)
        return np.linspace(min_pos, max_pos, self.n_points)

    def execute_linear_measure(self, mcl, exp_pos=None):
        if exp_pos is None:
            exp_pos = self._get_common_position_range()

        self.expected_pos = exp_pos
        self.measured_pos = np.zeros(self.n_points)

        cmd_flat = np.zeros(self.number_of_actuators)
        cmd = np.zeros(self.number_of_actuators)

        for idx in range(self.n_points):

            cmd = np.zeros(self.number_of_actuators)
            self._bmc.set_shape(cmd_flat)
            wf_flat = self._wyko.wavefront(timeout_in_sec=10)

            cmd[self.act] = mcl._sampled_p2c(self.act, exp_pos[idx])
            self._bmc.set_shape(cmd)
            wf_meas = self._wyko.wavefront(timeout_in_sec=10)
            wf_sub = wf_meas - wf_flat
            wf_sub = wf_sub - np.ma.median(wf_sub)
            self._bmc.set_shape(cmd_flat)
            self.measured_pos[idx] = self._max_wavefront(wf_sub)

        return self.measured_pos

    def execute_multiple_linear_measure(self, mcl, Ntimes=10, exp_pos=None):

        self.measured_pos_vector = np.zeros((Ntimes, self.n_points))
        for t in range(Ntimes):
            print('time%d' % t)
            self.measured_pos_vector[t] = self.execute_linear_measure(
                mcl, exp_pos)

    def save_linear_results(self, n_scan, fname):
        hdr = fits.Header()
        hdr['NSCAN'] = n_scan
        hdr['ACT'] = self.act
        fits.writeto(fname, self.expected_pos, hdr)
        fits.append(fname, self.measured_pos_vector)

    @staticmethod
    def load_linear_results(fname):
        header = fits.getheader(fname)
        hduList = fits.open(fname)
        expected_pos = hduList[0].data
        measured_pos_vector = hduList[1].data
        n_scan = header['NSCAN']
        act = header['ACT']
        return expected_pos, measured_pos_vector, n_scan, act

    def _max_wavefront(self, wf):
        coord_max = np.argwhere(np.abs(wf) == np.max(np.abs(wf)))[0]
        y, x = coord_max[0], coord_max[1]
        list_to_avarage = []
        # avoid masked data
        for yi in range(y - 1, y + 2):
            for xi in range(x - 1, x + 2):
                if(wf[yi, xi].data != 0.):
                    list_to_avarage.append(wf[yi, xi])
        list_to_avarage = np.array(list_to_avarage)
        # return wf[y, x]
        return np.median(list_to_avarage)

    @property
    def number_of_actuators(self):
        return self._bmc.get_number_of_actuators()

    @property
    def selected_actuator(self):
        return self.act

    @property
    def get_num_of_voltage_scans_list(self):
        return self.scan_list


class LinearityResponseAnalyzer():

    def __init__(self, fname):
        self.expected_pos, self.measured_pos_vector, self.n_scan, self.act = InterpolationErrorAnalyzer.load_linear_results(
            fname)

    def show_linearity(self):
        import matplotlib.pyplot as plt

        x = self.get_expected_deflections / 1e-9
        y = self.get_mean_deflections / 1e-9
        yerr = self.get_std_deflections / 1e-9

        plt.figure()
        plt.clf()
        plt.plot(x, y, 'b.', markersize=0.5, label='data')
        plt.errorbar(x, y, 10 * yerr, ls=None,
                     fmt='.', markersize=0.5, label='$10\sigma$')
        plt.xlabel('$x_{exp} [nm]$', size=15)
        plt.ylabel('$x_{meas} [nm]$', size=15)
        plt.title('Actuator %d' % self.act, size=15)
        plt.grid()

        a, b, chisq = self.execute_fit()

        def func(data, a, b):
            return a * data + b
        plt.plot(x, func(x, a[0], a[1]), '-r', lw=0.5, label='fit')
        plt.legend(loc='best')

    def show_data(self):
        import matplotlib.pyplot as plt

        x = self.get_expected_deflections / 1e-9
        y = self.measured_pos_vector

        plt.figure()
        plt.clf()
        for idx in range(y.shape[0]):
            plt.plot(x, y[idx] / 1e-9, '.', label='data')
        plt.xlabel('$x_{exp} [nm]$', size=15)
        plt.ylabel('$x_{meas} [nm]$', size=15)
        plt.title('Actuator %d' % self.act, size=15)
        plt.grid()

    def show_meas_vs_exp(self):
        import matplotlib.pyplot as plt

        x = self.get_expected_deflections / 1e-9
        y = self.measured_pos_vector

        plt.figure()
        plt.clf()
        for idx in range(y.shape[0]):
            plt.plot(x, y[idx] / 1e-9 - x, '.-', label='data act%d' % self.act)
        plt.xlabel('$x_{exp} [nm]$', size=15)
        plt.ylabel('$x_{meas} - x_{exp} [nm]$', size=15)
        plt.title('Actuator %d' % self.act, size=15)
        plt.grid()

    def execute_fit(self):
        x = self.get_expected_deflections / 1e-9
        y = self.get_mean_deflections / 1e-9
        yerr = self.get_std_deflections / 1e-9
        # yerr = (self.measured_pos_vector.max(axis=0) -
        #         self.measured_pos_vector.min(axis=0)) * 0.5

        def func(data, a, b):
            return a * data + b

        par, cov = curve_fit(
            func, x, y, p0=[1., 1.], sigma=yerr)
        res = y - func(x, par[0], par[1])
        chisq = sum((res / yerr)**2)

        return par, cov, chisq

    def show_meas_vs_fit(self):
        import matplotlib.pyplot as plt

        a, b, chisq = self.execute_fit()
        x = self.get_expected_deflections / 1e-9
        y = self.get_mean_deflections / 1e-9
        yerr = self.get_std_deflections / 1e-9

        def func(data, a, b):
            return a * data + b
        plt.figure()
        plt.clf()
        plt.errorbar(x, y - x, yerr, fmt='o', ls=None, label='$\sigma$')
        plt.plot(x, y - func(x, a[0], a[1]), 'or', label='fitting residual')
        plt.legend(loc='best')
        plt.grid()
        plt.xlabel('$x_{exp}$\t [nm]', size=15)
        plt.ylabel('$\epsilon_{x}$\t [nm]', size=15)
        # for idx in range(self.measured_pos_vector.shape[0]):
        #     plt.plot(
        # x, self.measured_pos_vector[idx] / 1e-9 - x, '.', label='data')

    @property
    def get_mean_deflections(self):
        return self.measured_pos_vector.mean(axis=0)

    @property
    def get_std_deflections(self):
        return self.measured_pos_vector.std(axis=0)

    @property
    def get_expected_deflections(self):
        return self.expected_pos

    @property
    def get_actuator(self):
        return self.act
