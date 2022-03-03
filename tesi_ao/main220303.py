from tesi_ao import sandbox
import matplotlib.pyplot as plt
import numpy as np


class InterpolationErrorAnalyzer():

    ACTUATOR = 63
    NUM_SCAN_LIST = [10, 20, 30, 40, 50]
    fpath = 'prova/act63/main220303/cplm'
    ffmt = '.fits'

    def execute_measure(self, fname, Nscan):

        act_list = [self.ACTUATOR]

        wyko, bmc = sandbox.create_devices()

        cplm = sandbox.CommandToPositionLinearizationMeasurer(wyko, bmc)

        cplm.NUMBER_STEPS_VOLTAGE_SCAN = Nscan

        cplm.execute_command_scan(act_list)

        cplm.save_results(fname)

    def get_mcl_from_file(self, fname):

        cpla = sandbox.CommandToPositionLinearizationAnalyzer(fname)
        mcl = cpla.compute_linearization()

        return mcl

    def plot_interpolation_function(self, mcl):

        # plt.title('act#%: interpolation functions for several scans' %
        #         self.ACTUATOR)
        #plt.xlabel('Commands [au]')
        #plt.ylabel('Deflection [m]')

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
            self.execute_measure(fname, scans)

    def load_mcls(self, version_file):

        mcl_list = []

        for scans in self.NUM_SCAN_LIST:
            fname = self.fpath + '%d' % scans + version_file + self.ffmt
            mcl_list.append(self.get_mcl_from_file(fname))

        return mcl_list

    def plot_all_interpolation_functions(self, mcl_list):
        for mcl in mcl_list:
            self.plot_interpolation_function(mcl)

    def plot_interpolation_error(self, mcl_list):
        Npt = 1024

        min_container = []
        max_container = []

        for mcl in mcl_list:  # TypeError:cannot unpack non-iterable MemsCommandLinearization object

            min_container.append(min(mcl._deflection[0]))

            max_container.append(max(mcl._deflection[0]))

        common_span_deflections = np.linspace(
            max(min_container), min(max_container), Npt)

        f_ref = mcl_list[-1]._finter[0]  # interp func with the biggest #scans

        for mcl in mcl_list:
            f_i = mcl._finter[0]
            for scans in self.NUM_SCAN_LIST:
                plt.plot(common_span_deflections, f_i(common_span_deflections) -
                         f_ref(common_span_deflections), '.-', label='%d scans' % scans)
