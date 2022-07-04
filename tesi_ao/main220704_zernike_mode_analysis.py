import numpy as np
from tesi_ao.mems_zernike_mode_analyzer import MemsAmplitudeLinearityAnalizer


def do_comparison(amp_idx):
    import matplotlib.pyplot as plt
    ffolder1 = '220629'
    ffolder2 = '220701a'

    main1 = Main220704(ffolder1)
    main2 = Main220704(ffolder2)

    plt.figure()
    plt.plot(main1.j_index_list, np.abs(
        main1.zs_measured_fitting_error[:, amp_idx] / main1.zs_expected_amplitudes[:, amp_idx]), '*')
    plt.plot(main2.j_index_list, np.abs(
        main2.zs_measured_fitting_error[:, amp_idx] / main2.zs_expected_amplitudes[:, amp_idx]), 's')
    plt.title('$a_{expected} = %g $m' %
              main1.zs_expected_amplitudes[0, amp_idx], size=10)
    plt.xlabel('j index', size=10)
    plt.ylabel('relative fitting error', size=10)


class Main220704():
    def __init__(self, ffolder):
        self.j_index_list = np.arange(2, 51)
        self.n_of_meas_amp = 24
        self.fpath = 'prova/mems_mode_measurements/' + ffolder + '/'
        self.zs_measured_amplitudes, self.zs_expected_amplitudes, self.zs_measured_fitting_error, self.clipped_amplitudes_list = self._collect_zernike_measurments()

    def _collect_zernike_measurments(self):
        # fpath = 'prova/mems_mode_measurements/220629/'
        num_of_measured_zernike = len(self.j_index_list)
        # n_total_acts = 140
        # frame_shape = (486, 640)
        zs_measured_amplitudes = np.ma.zeros(
            (num_of_measured_zernike, self.n_of_meas_amp))
        zs_expected_amplitudes = np.zeros(
            (num_of_measured_zernike, self.n_of_meas_amp))
        zs_measured_fitting_error = np.zeros(
            (num_of_measured_zernike, self.n_of_meas_amp))
        clipped_amplitudes_list = []

        for idx, j in enumerate(self.j_index_list):
            fname = self.fpath + 'z%d_act80.fits' % int(j)
            male_j = MemsAmplitudeLinearityAnalizer(fname)
            zs_measured_amplitudes[idx] = male_j.get_measured_amplitudes()
            zs_measured_fitting_error[idx] = male_j.get_measured_fitting_error(
            )
            clipped_amplitudes_list.append(
                male_j.get_clipped_modes_index_list())
            zs_expected_amplitudes[idx] = male_j.get_expected_amplitudes()
        return zs_measured_amplitudes, zs_expected_amplitudes, zs_measured_fitting_error, clipped_amplitudes_list

    def show_fitting_error_pattern_for_that_amplitude(self, amp_idx):
        import matplotlib.pyplot as plt
        # zs_measured_amplitudes, zs_expected_amplitudes, zs_measured_fitting_error, clipped_amplitudes_list = main_read_collected_zernike_measurments()
        # j_modes = np.arange(2, 51)

        plt.figure()
        plt.plot(self.j_index_list, np.abs(
            self.zs_measured_fitting_error[:, amp_idx] / self.zs_expected_amplitudes[:, amp_idx]), '.')
        plt.title('$a_{expected} = %g $m' %
                  self.zs_expected_amplitudes[0, amp_idx], size=10)
        plt.xlabel('j index', size=10)
        plt.ylabel('relative fitting error', size=10)

    def _show_for_that_amplitudes(self):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.clf()
        plt.xlabel('j index', size=10)
        plt.ylabel('relative fitting error', size=10)
        amp_list = self.zs_expected_amplitudes[0, 12:]
        for idx, a in enumerate(amp_list):
            plt.plot(self.j_index_list, np.abs(
                self.zs_measured_fitting_error[:, 12 + idx] / self.zs_expected_amplitudes[:, 12 + idx]), '.', label='$a_{exp} = %g$ m' % a)

    def _show_histogram_that_amplitudes(self):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.clf()
        plt.xlabel('j index', size=10)
        plt.ylabel('relative fitting error', size=10)
        amp_list = self.zs_expected_amplitudes[0, 12:]
        for idx, a in enumerate(amp_list):
            plt.bar(self.j_index_list, np.abs(
                self.zs_measured_fitting_error[:, 12 + idx] / self.zs_expected_amplitudes[:, 12 + idx]), label='$a_{exp} = %g$ m' % a)
