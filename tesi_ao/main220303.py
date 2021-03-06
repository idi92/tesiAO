from tesi_ao import main220316
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from tesi_ao.mems_command_to_position_linearization_measurer import CommandToPositionLinearizationMeasurer
from tesi_ao.mems_command_to_position_linearization_analyzer import CommandToPositionLinearizationAnalyzer


def _what_I_do_on_terminal():  # don't use!
    '''
    an example of how I used main220303
    '''
    iea = InterpolationErrorAnalyzer(ACTUATOR=63, NUM_SCAN_LIST=[
                                     10, 20, 30, 40, 50, 60, 100], test_points=20)

    # do and save WF maps for each scan sampling
    iea.do_more_scans('_f0')
    # load mcl objects into a list
    mcls_int = iea.load_mcls('_f0')
    # plot all the interpolated functions
    iea.plot_all_interpolation_functions(mcls_int)
    # plot interpolated function difference wrt the one with the biggest
    # samples
    iea.plot_interpolation_difference(mcls_int)
    # from the 'old' mcls elements, we need the interpolated functions
    # to compute p2c and save the a 'new' measured mcl object
    iea.do_calibrated_measure(mcls_int, '_f0')
    # load new mcl
    mcls_meas = iea.load_calibrated_measure('_f0')
    # Plot the difference between the measured and expected deflection, as a
    # function of the expected one
    rms_list = iea.plot_Measured_vs_Expected_common(mcls_meas, mcls_int)
    iea.fitting_Meas_vs_Exp_common(mcls_meas, rms_list, mcls_int)


class InterpolationErrorAnalyzer(object):

    ffmt = '.fits'

    def __init__(self, ACTUATOR=63, NUM_SCAN_LIST=[10, 20, 30, 40, 50, 60, 100], test_points=10):
        self.ACTUATOR = ACTUATOR
        self.NUM_SCAN_LIST = NUM_SCAN_LIST
        self.test_points = test_points
        self.fpath = 'prova/act%d' % ACTUATOR + '/main220303/cplm'

    def _execute_measure(self, fname, Nscan):
        '''
        Executes WF maps measure, one for each scan, and saves the
        related CPLM object into fname.fits.  
        '''
        act_list = [self.ACTUATOR]

        wyko, bmc = main220316.create_devices()

        cplm = CommandToPositionLinearizationMeasurer(wyko, bmc)

        cplm.NUMBER_STEPS_VOLTAGE_SCAN = Nscan

        cplm.execute_command_scan(act_list)

        cplm.save_results(fname)

    def _get_mcl_from_file(self, fname):
        '''
        From a fits file, loads CPLA object and evaluating 
        interpolation function.
        Returns the related MemsCommandLinearization object.
        '''
        cpla = CommandToPositionLinearizationAnalyzer(fname)
        mcl = cpla.compute_linearization()

        return mcl

    def _plot_interpolation_function(self, mcl):

        plt.plot(mcl._cmd_vector[0], mcl._deflection[0], 'o',
                 label='%d scans' % mcl._cmd_vector.shape[1])

        Npt = 1024

        f_int = mcl._finter[0]

        span = np.linspace(
            min(mcl._cmd_vector[0]), max(mcl._cmd_vector[0]), Npt)

        plt.plot(span, f_int(span), '-', color=plt.gca().lines[-1].get_color())

    def _get_common_cmds_range(self, mcl_list):
        '''
        Returns the extremes[a,b] of the common cmd domain
        between all interpolated functions
        Input: list, mcl_list
        Returns: a, b
        '''
        min_container = []
        max_container = []
        for mcl in mcl_list:
            min_container.append(min(mcl._cmd_vector[0]))
            max_container.append(max(mcl._cmd_vector[0]))
        a = max(min_container)
        b = min(max_container)
        return a, b

    def _get_common_deflections_range(self, mcl_list):
        min_container = []
        max_container = []
        for mcl in mcl_list:
            min_container.append(min(mcl._calibrated_position[0]))
            max_container.append(max(mcl._calibrated_position[0]))
        a = max(min_container)
        b = min(max_container)
        return a, b

    def do_more_scans(self, version_file):
        '''
        For each scan sampling defined in NUM_SCAN_LIST, 
        executes WF mapping through the class objects CPLM and CPLA defined in sandbox.py,
        and saves into file.fits        
        '''
        for scans in self.NUM_SCAN_LIST:
            print('\n%d voltage scans:' % scans)
            fname = self.fpath + '%d' % scans + version_file + self.ffmt
            self._execute_measure(fname, scans)

    def load_mcls(self, version_file):
        '''
        Loads MemsCommandLinearization objects defined in sandbox.py,
        computed by do_more_scans
        and returns them into a list.  
        Input: string,'vesion_file'
        Return: list, mcl_list 
        len(mcl_list) == number of interpolated function (one for each scan sampling)
        '''
        mcl_list = []

        for scans in self.NUM_SCAN_LIST:
            fname = self.fpath + '%d' % scans + version_file + self.ffmt
            mcl_list.append(self._get_mcl_from_file(fname))

        return mcl_list

    def plot_all_interpolation_functions(self, mcl_list):
        '''
        Plots all interpolated functions obtained by varying scan sampling,
        as a function of actuator's deflections.
        '''
        plt.figure()
        plt.clf()
        plt.ion()
        plt.title('act#%d: interpolation functions for several scans' %
                  self.ACTUATOR, size=25)
        for mcl in mcl_list:
            self._plot_interpolation_function(mcl)

        plt.xlabel('Commands [au]', size=25)
        plt.ylabel('Deflection [m]', size=25)
        plt.grid()
        plt.legend(loc='best')

    def plot_interpolation_difference(self, mcl_list):
        '''
        Plots the difference between all the interpolated function with
        respect to the one computed with the biggest scan sampling, as a function of
        actuators deflections.
        Input: list, mcl_list 
        '''
        Npt = 1024

        # looking for the common deflections domain for the interpolated
        # functions
        min_span, max_span = self._get_common_cmds_range(mcl_list)

        common_span_cmds = np.linspace(
            min_span, max_span, Npt)

        # interpolated function with the biggest scans sampling
        f_ref = mcl_list[-1]._finter[0]

        plt.figure()
        plt.clf()
        plt.ion()
        plt.title('act#%d:' % self.ACTUATOR +
                  'cubic interpolation error w-r-t %dscans' % max(self.NUM_SCAN_LIST), size=25)

        for idx, scans in enumerate(self.NUM_SCAN_LIST):
            f_i = mcl_list[idx]._finter[0]
            plt.plot(common_span_cmds, (f_i(common_span_cmds) -
                                        f_ref(common_span_cmds)) / 1e-9, '.-', label='%d scans' % scans)
            print((f_i(common_span_cmds) - f_ref(common_span_cmds)).std())

        plt.legend(loc='best')
        plt.grid()
        plt.ylabel('Deflection Difference [m]', size=25)
        plt.xlabel('cmd [au]', size=25)

    def do_calibrated_measure(self, mcl_list, version):
        '''
        Though the interpolated functions contained in the 'old' MCL objects 
        and listed in mcl_list, saves new WF maps using converted
        actuator's deflections (calling p2c and MyCalibrationMeasurer class as defined below).
        Input:
        list, mcl_list
        string, 'file version'
        '''
        Npt = self.test_points
        # self.NUM_SCAN_LIST

        act_list = [self.ACTUATOR]

        wyko, bmc = main220316.create_devices()

        min_span, max_span = self._get_common_deflections_range(mcl_list)
        expected_deflection = np.linspace(max_span, min_span, Npt)

        # expected_deflection = np.linspace(-800e-9, 1600e-9, Npt) #@act63

        #converted_cmd = np.zeros((len(mcl_list), Npt))

        for idx, mcl in enumerate(mcl_list):
            mcm = MyCalibrationMeasurer(wyko, bmc, mcl, expected_deflection)
            mcm.execute_command_scan(act_list)
            fname = self.fpath + '%d' % Npt + 'meas' + version + \
                '_cal%d' % self.NUM_SCAN_LIST[idx] + self.ffmt
            mcm.save_results(fname)

    def load_calibrated_measure(self, version):
        '''
        Loads the 'new' mcl objects from file created by do_calibrated_measure,
        and returns them into a list.
        Input: string, 'file_version'
        Return: list, mcl_list
        '''
        mcl_list = []
        Npt = self.test_points
        for scans in self.NUM_SCAN_LIST:
            fname = self.fpath + '%d' % Npt + 'meas' + version + \
                '_cal%d' % scans + self.ffmt

            mcl_list.append(self._get_mcl_from_file(fname))

        return mcl_list

    def plot_Measured_vs_Expected_common(self, mcl_meas, mcl_int):
        '''
        Plots the difference between the measured and expected deflection,
        as a function of the expected one. 
        mcl_meas[i]== element of the list loaded from load_calibrated_measure
        Input: list, mcls_meas
        list, mcl_int (used for common deflection domain evaluation)
        '''
        Npt = self.test_points

        plt.figure()
        plt.clf()
        min_span, max_span = self._get_common_deflection_range(mcl_int)
        #min_span = -800e-9
        #max_span = 1600e-9
        x_exp = np.linspace(min_span, max_span, Npt)  # expected deflections
        rms_list = []
        for idx in np.arange(len(mcl_meas)):

            x_obs = mcl_meas[idx]._deflection[0]
            y = x_obs - x_exp
            rms = y.std()
            rms = rms / 1.e-9
            rms_list.append(y.std())
            plt.plot(x_exp / 1.e-9, y / 1.e-9, 'o-', label='%d scans' %
                     self.NUM_SCAN_LIST[idx])
            print('rms = %g' % rms + 'nm\t' +
                  '(Sampling: %d scans)' % self. NUM_SCAN_LIST[idx])

        plt.legend(loc='best')
        plt.grid()
        plt.xlabel('$x_{exp} [nm]$', size=25)
        plt.ylabel('$x_{obs} - x_{exp} [nm]$', size=25)
        plt.title('act#%d:' % self.ACTUATOR +
                  ' Error in deflection cmds for each interpolation functions Common', size=25)
        return rms_list

        # something wrong
    def fitting_Meas_vs_Exp_common(self, mcl_meas, rms_list, mcl_int):
        '''
        Plots the best fits for measured vs expected deflection, for each scan sampling.
        '''
        Npt = self.test_points

        plt.figure()
        plt.clf()
        min_span, max_span = self._get_common_deflection_range(mcl_int)
        #min_span = -800e-9
        #max_span = 1600e-9
        x_exp = np.linspace(min_span, max_span, Npt)
        ones = np.ones(Npt)
        xx = np.linspace(min_span, max_span, 1024)
        for idx in np.arange(len(mcl_meas)):

            x_obs = mcl_meas[idx]._deflection[0]

            plt.plot(x_exp, x_obs, 'o', label='%d scans' %
                     self.NUM_SCAN_LIST[idx])
            sigma = ones * rms_list[idx]
            coeff, coeff_cov = np.polyfit(
                x_exp, x_obs, 1, w=sigma, cov=True, full=False)
            err_coeff = np.sqrt(np.diag(coeff_cov))
            print('\nFit relative to Sampling: %d scans)' %
                  self. NUM_SCAN_LIST[idx])
            print('A = %g' % coeff[0] + '\t+/- %g ' % err_coeff[0])
            print('offset = %g' % coeff[1] + '\t+/- %g' % err_coeff[1])
            print('Cov Matrix:')
            print(coeff_cov)
            fit_func = np.poly1d(coeff)
            residuals = x_obs - fit_func(x_obs)
            chi_2 = np.sum((residuals / sigma)**2)
            print('Chi2 = %g' % chi_2)
            dof = len(x_obs) - len(coeff)
            chi2red = chi_2 / float(dof)
            print('RedChi2 = %g' % chi2red)
            plt.plot(xx, fit_func(xx), '-', label='relative fit',
                     color=plt.gca().lines[-1].get_color())
            # plt.errorbar(x_exp, x_obs, sigma,
            #            color=plt.gca().lines[-1].get_color())
        plt.legend(loc='best')
        plt.grid()
        plt.xlabel('$x_{exp} [m]$', size=25)
        plt.ylabel('$x_{obs} [m]$', size=25)
        plt.title('act#%d:' % self.ACTUATOR +
                  ' Common deflection span', size=25)

# similar to CommandtoPositionLinearizationMeasurer


class MyCalibrationMeasurer(object):  # changes when bmc set shape
    '''
    As CommandToPositionLinearizationMeasurer defined in sandbox.py, 
    acquires WF maps, one for each expected deflection command. 
    These deflections are converted in voltage commands through 
    p2c function stored in MCL object.
    '''
    NUMBER_WAVEFRONTS_TO_AVERAGE = 1
    #NUMBER_STEPS_VOLTAGE_SCAN = 10

    def __init__(self, interferometer, mems_deformable_mirror, mlc, expected_deflections):
        self._interf = interferometer
        self._bmc = mems_deformable_mirror
        self._n_acts = self._bmc.get_number_of_actuators()
        self._mlc = mlc
        self._exp_deflections = expected_deflections
        self.NUMBER_STEPS_VOLTAGE_SCAN = len(expected_deflections)
        self._wfflat = None

    def _get_zero_command_wavefront(self):
        if self._wfflat is None:
            cmd = np.zeros(self._n_acts)
            self._bmc.set_shape(cmd)
            self._wfflat = self._interf.wavefront(
                self.NUMBER_WAVEFRONTS_TO_AVERAGE)
        return self._wfflat

    def execute_command_scan(self, act_list=None):
        if act_list is None:
            act_list = np.arange(self._n_acts)

        self._actuators_list = np.array(act_list)
        n_acts_to_meas = len(self._actuators_list)

        wfflat = self._get_zero_command_wavefront()

        self._reference_cmds = self._bmc.get_reference_shape()
        self._reference_tag = self._bmc.get_reference_shape_tag()

        self._cmd_vector = np.zeros((n_acts_to_meas,
                                     self.NUMBER_STEPS_VOLTAGE_SCAN))

        self._wfs = np.ma.zeros(
            (n_acts_to_meas, self.NUMBER_STEPS_VOLTAGE_SCAN,
             wfflat.shape[0], wfflat.shape[1]))
        N_pixels = self._wfs.shape[2] * self._wfs.shape[3]

        for act_idx, act in enumerate(self._actuators_list):

            self._cmd_vector[act_idx] = self._mcl.linear_p2c(
                act, self._exp_deflections)
            for cmd_idx in range(len(self._cmd_vector[act_idx])):
                print("Act:%d - command" % (act))
                cmd = np.zeros(self._n_acts)
                cmd[act] = self._mlc.linear_p2c(
                    act, self._exp_deflections[cmd_idx])
                self._bmc.set_shape(cmd)
                self._wfs[act_idx, cmd_idx, :,
                          :] = self._get_wavefront_flat_subtracted()
                masked_pixels = self._wfs[act_idx, cmd_idx].mask.sum()
                masked_ratio = masked_pixels / N_pixels
                if masked_ratio > 0.8227:
                    print('Warning: Bad measure acquired for: act%d' %
                          act_idx + ' cmd_idx %d' % cmd_idx)
                    self._avoid_saturated_measures(
                        masked_ratio, act_idx, cmd_idx, N_pixels)

    def _avoid_saturated_measures(self, masked_ratio, act_idx, cmd_idx, N_pixels):

        while masked_ratio > 0.8227:
            self._wfs[act_idx, cmd_idx, :,
                      :] = self._get_wavefront_flat_subtracted()
            masked_pixels = self._wfs[act_idx, cmd_idx].mask.sum()
            masked_ratio = masked_pixels / N_pixels

        print('Repeated measure completed!')

    def _get_wavefront_flat_subtracted(self):
        dd = self._interf.wavefront(
            self.NUMBER_WAVEFRONTS_TO_AVERAGE) - self._get_zero_command_wavefront()
        return dd - np.ma.median(dd)

    def _reset_flat_wavefront(self):
        self._wfflat = None

    def save_results(self, fname):
        hdr = fits.Header()
        hdr['REF_TAG'] = self._reference_tag
        hdr['N_AV_FR'] = self.NUMBER_WAVEFRONTS_TO_AVERAGE
        fits.writeto(fname, self._wfs.data, hdr)
        fits.append(fname, self._wfs.mask.astype(int))
        fits.append(fname, self._cmd_vector)
        fits.append(fname, self._actuators_list)
        fits.append(fname, self._reference_cmds)

    @staticmethod
    def load(fname):
        header = fits.getheader(fname)
        hduList = fits.open(fname)
        wfs_data = hduList[0].data
        wfs_mask = hduList[1].data.astype(bool)
        wfs = np.ma.masked_array(data=wfs_data, mask=wfs_mask)
        cmd_vector = hduList[2].data
        actuators_list = hduList[3].data
        reference_commands = hduList[4].data
        return {'wfs': wfs,
                'cmd_vector': cmd_vector,
                'actuators_list': actuators_list,
                'reference_shape': reference_commands,
                'reference_shape_tag': header['REF_TAG']
                }
