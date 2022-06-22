import numpy as np
from tesi_ao import main220316
from tesi_ao.mems_command_linearization import MemsCommandLinearization
from tesi_ao.mems_interpolation_error import InterpolationErrorAnalyzer, LinearityResponseAnalyzer
from tesi_ao.mems_cross_talk import CrossTalkAnalyzer
from tesi_ao.mems_flat_stability import FlatStabilityAnalyzer
from tesi_ao.mems_flat_reshaper import MemsFlatReshaper


class Chap3():
    mcl_fname = 'prova/misure_ripetute/mcl_all_def.fits'

    def __init__(self):
        self.mcl = MemsCommandLinearization.load(self.mcl_fname)

    def show_normal_flat220616(self):
        import matplotlib.pyplot as plt
        fsa_name = 'prova/flat_stability/normal_flat220616.fits'
        wf_mean, wf_sigma, n_meas = FlatStabilityAnalyzer.load_map(fsa_name)
        plt.figure()
        plt.clf()
        plt.imshow(wf_mean / 1e-9)
        plt.colorbar(label='[nm]')
        plt.figure()
        plt.clf()
        plt.imshow(wf_sigma / 1e-9)
        plt.colorbar(label='[nm]')

    def show_deformed_flat(self):
        import matplotlib.pyplot as plt
        fsa_name = 'prova/flat_stability/deformed_flat220606.fits'
        wf_mean, wf_sigma, n_meas = FlatStabilityAnalyzer.load_map(fsa_name)
        plt.figure()
        plt.clf()
        plt.imshow(wf_mean / 1e-9)
        plt.colorbar(label='[nm]')
        plt.figure()
        plt.clf()
        plt.imshow(wf_sigma / 1e-9)
        plt.colorbar(label='[nm]')

    def show_increasing_sampling_measuraments_for_act(self, act):
        iea = InterpolationErrorAnalyzer(act)
        mcl_list = iea.load_multiple_scans_interpolation()
        iea.show_all_interpolation_functions(mcl_list)

    def show_interpolation_difference_for_act(self, act):
        iea = InterpolationErrorAnalyzer(act)
        mcl_list = iea.load_multiple_scans_interpolation()
        iea.show_interpolation_difference(mcl_list)

    def show_deflection_difference_for_act(self, act):
        fname = 'prova/mems_interpolation_error/dati_per_fit_act%d' % act + \
            '_nscan20_times100.fits'
        lra = LinearityResponseAnalyzer(fname)
        lra.show_meas_vs_exp()

    def show_linear_fit_for_act(self, act):
        fname = 'prova/mems_interpolation_error/dati_per_fit_act%d' % act + \
            '_nscan20_times100.fits'
        lra = LinearityResponseAnalyzer(fname)
        lra.show_linearity()
        return lra.execute_fit()

    def show_residual_comparison_for_act(self, act):
        fname = 'prova/mems_interpolation_error/dati_per_fit_act%d' % act + \
            '_nscan20_times100.fits'
        lra = LinearityResponseAnalyzer(fname)
        lra.show_meas_vs_fit()

    def show_all_calibration_curves(self):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.clf()
        mcl = self.mcl
        for idx, act in enumerate(mcl._actuators_list):
            a = np.min(mcl._cmd_vector[act])
            b = np.max(mcl._cmd_vector[act])
            vv = np.linspace(a, b, 1000)
            plt.plot(vv, mcl._finter[act](vv) / 1.e-9, '-', label='finter')
        plt.xlabel('Command [au]', size=15)
        plt.ylabel('Deflection [nm]', size=15)
        plt.title('Calibration curve per actuator', size=15)
        plt.grid()

    def show_inner_and_outer_calibration_curves_alone(self):
        import matplotlib.pyplot as plt
        mcl = self.mcl
        list1 = np.arange(0, 10)
        list2 = np.arange(10, 130, 12)
        list3 = np.arange(21, 141, 12)
        list4 = np.arange(130, 140)
        ls = []
        ls.append(list1)
        ls.append(list2)
        ls.append(list3)
        ls.append(list4)
        act_bordo = np.array(ls).ravel()
        all_act = np.arange(140)
        other_act = np.delete(all_act, act_bordo)
        plt.figure()
        plt.clf()
        plt.title('Outer actuators', size=15)
        plt.xlabel('cmd [au]', size=15)
        plt.ylabel('pos [nm]', size=15)
        for act in act_bordo:
            plt.plot(mcl._cmd_vector[act], mcl._deflection[act] / 1e-9, '-')
        plt.grid()
        plt.figure()
        plt.clf()
        plt.title('Inner actuators', size=15)
        plt.xlabel('cmd [au]', size=15)
        plt.ylabel('pos [nm]', size=15)
        for act in other_act:
            plt.plot(mcl._cmd_vector[act], mcl._deflection[act] / 1e-9, '-')
        plt.grid()

    def show_crosstalk_next_to_act(self, act=76, pos_str='1250nm'):
        fname = 'prova/misure_crosstalk/act76_pos' + pos_str + '.fits'
        cta = CrossTalkAnalyzer(fname)
        cta.show_crosstalk_along_x_axis()
        cta.show_crosstalk_along_y_axis()
        cta.measure_crosstalk()

    def show_flattening_results(self):
        import matplotlib.pyplot as plt
        fname = 'prova/misure_con_tappo/misure_spianamento/dati_spianamento_per_plot.fits'
        flatten_data, pos_list, thres_list, n_points, n_flatten = MemsFlatReshaper.load_flatten_data(
            fname)
        col = ['k', 'g', 'r', 'b', 'c']
        lstyle = ['.', '+', '*']
        plt.figure()
        plt.clf()

        for i, thres in enumerate(thres_list):
            for j, pos in enumerate(pos_list):
                plt.plot(flatten_data[i, j, 0, :] / 1e-9, '-' +
                         col[i] + lstyle[j], lw=0.2, label='thres = %g' % thres + ' start_wf = %g nm' % pos)
        plt.legend(loc='best')
        plt.xlabel('Iterations', size=15)
        plt.label('WF amplitude rms [nm]', size=15)
