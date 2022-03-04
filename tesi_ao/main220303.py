from tesi_ao import sandbox
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


class InterpolationErrorAnalyzer():

    ACTUATOR = 63
    NUM_SCAN_LIST = [10, 20, 30, 40, 50, 60, 100]
    test_points = 10
    fpath = 'prova/act%d' % ACTUATOR + '/main220303/cplm'
    ffmt = '.fits'

    def _execute_measure(self, fname, Nscan):

        act_list = [self.ACTUATOR]

        wyko, bmc = sandbox.create_devices()

        cplm = sandbox.CommandToPositionLinearizationMeasurer(wyko, bmc)

        cplm.NUMBER_STEPS_VOLTAGE_SCAN = Nscan

        cplm.execute_command_scan(act_list)

        cplm.save_results(fname)

    def _get_mcl_from_file(self, fname):

        cpla = sandbox.CommandToPositionLinearizationAnalyzer(fname)
        mcl = cpla.compute_linearization()

        return mcl

    def _plot_interpolation_function(self, mcl):

        plt.plot(mcl._cmd_vector[0], mcl._deflection[0], 'o',
                 label='%d scans' % mcl._cmd_vector.shape[1])

        Npt = 1024

        f_int = mcl._finter[0]

        span = np.linspace(
            min(mcl._deflection[0]), max(mcl._deflection[0]), Npt)

        plt.plot(f_int(span), span, '-', color=plt.gca().lines[-1].get_color())

    def do_more_scans(self, version_file):

        for scans in self.NUM_SCAN_LIST:
            print('\n%d voltage scans:' % scans)
            fname = self.fpath + '%d' % scans + version_file + self.ffmt
            self._execute_measure(fname, scans)

    def load_mcls(self, version_file):
        '''
        Loads MemsCommandLinarization objects in a list
        '''
        mcl_list = []

        for scans in self.NUM_SCAN_LIST:
            fname = self.fpath + '%d' % scans + version_file + self.ffmt
            mcl_list.append(self._get_mcl_from_file(fname))

        return mcl_list

    def plot_all_interpolation_functions(self, mcl_list):

        plt.figure(101)
        plt.clf()
        plt.ion()
        plt.title('act#%d: interpolation functions for several scans' %
                  self.ACTUATOR)
        for mcl in mcl_list:
            self._plot_interpolation_function(mcl)

        plt.xlabel('Commands [au]')
        plt.ylabel('Deflection [m]')
        plt.grid()
        plt.legend(loc='best')

    def plot_interpolation_error(self, mcl_list):

        Npt = 1024

        min_container = []
        max_container = []

        for mcl in mcl_list:

            min_container.append(min(mcl._deflection[0]))

            max_container.append(max(mcl._deflection[0]))

        common_span_deflections = np.linspace(
            max(min_container), min(max_container), Npt)

        f_ref = mcl_list[-1]._finter[0]  # interp func with the biggest #scans

        plt.figure(102)
        plt.clf()
        plt.ion()
        plt.title('act#%d:' % self.ACTUATOR +
                  'cubic interpolation error w-r-t %dscans' % max(self.NUM_SCAN_LIST))

        for idx, scans in enumerate(self.NUM_SCAN_LIST):
            f_i = mcl_list[idx]._finter[0]
            plt.plot(common_span_deflections, f_i(common_span_deflections) -
                     f_ref(common_span_deflections), '.-', label='%d scans' % scans)

        plt.legend(loc='best')
        plt.grid()
        plt.ylabel('Command Difference [au]')
        plt.xlabel('Deflection [m]')

    def do_calibrated_measure(self, mcl_list, version):

        Npt = self.test_points
        self.NUM_SCAN_LIST

        act_list = [self.ACTUATOR]

        wyko, bmc = sandbox.create_devices()

        expected_deflection = np.linspace(-800e-9, 1600e-9, Npt)

        converted_cmd = np.zeros((len(mcl_list), Npt))

        for idx, mcl in enumerate(mcl_list):
            mcm = MyCalibrationMeasurer(wyko, bmc, mcl)
            mcm.execute_command_scan(act_list)
            fname = self.fpath + '%d' % Npt + 'meas' + version + \
                '_cal%d' % self.NUM_SCAN_LIST[idx] + self.ffmt
            mcm.save_results(fname)

    def load_calibrated_measure(self, version):

        mcl_list = []
        Npt = self.test_points
        for scans in self.NUM_SCAN_LIST:
            fname = self.fpath + '%d' % Npt + 'meas' + version + \
                '_cal%d' % scans + self.ffmt
            mcl_list.append(self._get_mcl_from_file(fname))

        return mcl_list

    def plot_Measured_vs_Expected(self, mcl_meas):

        Npt = self.test_points

        plt.figure(456)
        plt.clf()
        x_exp = np.linspace(-800e-9, 1600e-9, Npt)
        for idx in np.arange(len(mcl_meas)):

            x_obs = mcl_meas[idx]._deflection[0]
            y = x_obs - x_exp

            plt.plot(x_exp, y, 'o-', label='%d scans' %
                     self.NUM_SCAN_LIST[idx])

        plt.legend(loc='best')
        plt.grid()
        plt.xlabel('$x_{exp} [m]$')
        plt.ylabel('$x_{obs} - x_{exp} [m]$')
        plt.title('act#%d:' % self.ACTUATOR +
                  ' Error in deflection cmds for each interpolation functions')


# similar to CommandtoPositionLinearizationMeasurer
class MyCalibrationMeasurer(object):  # changes when bmc set shape

    NUMBER_WAVEFRONTS_TO_AVERAGE = 1
    NUMBER_STEPS_VOLTAGE_SCAN = 10

    def __init__(self, interferometer, mems_deformable_mirror, mlc):
        self._interf = interferometer
        self._bmc = mems_deformable_mirror
        self._n_acts = self._bmc.get_number_of_actuators()
        self._mlc = mlc
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

        expected_deflection = np.linspace(-800e-9,
                                          1600e-9, self.NUMBER_STEPS_VOLTAGE_SCAN)
        for act_idx, act in enumerate(self._actuators_list):

            self._cmd_vector[act_idx] = self._mlc.p2c(act, expected_deflection)
            for cmd_idx, cmdi in enumerate(self._cmd_vector[act_idx]):
                print("Act:%d - command %g" % (act, cmdi))
                cmd = np.zeros(self._n_acts)
                cmd[act] = cmdi
                self._bmc.set_shape(cmd)
                self._wfs[act_idx, cmd_idx, :,
                          :] = self._get_wavefront_flat_subtracted()

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
