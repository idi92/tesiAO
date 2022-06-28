import numpy as np
from tesi_ao import main220316
from tesi_ao.mems_command_linearization import MemsCommandLinearization
from tesi_ao.mems_interpolation_error import InterpolationErrorAnalyzer, LinearityResponseAnalyzer
from tesi_ao.mems_cross_talk import CrossTalkAnalyzer
from tesi_ao.mems_flat_stability import FlatStabilityAnalyzer
from tesi_ao.mems_flat_reshaper import MemsFlatReshaper
from tesi_ao.mems_max_stoke_coupling import CouplingMeasurer
from tesi_ao.mems_display import Boston140Display


class Chap3():
    mcl_fname = 'prova/misure_ripetute/mcl_all_def.fits'

    def __init__(self):
        self.mcl = MemsCommandLinearization.load(self.mcl_fname)

    def show_normal_flat220616(self):
        import matplotlib.pyplot as plt
        fsa = FlatStabilityAnalyzer()
        fsa.load_flat_220616()
        # levo outlier
        #fsa.wfs_flat.mask[14:16] = True
        amp_i = fsa.wfs_flat.std(axis=(1, 2))
        plt.figure()
        plt.clf()
        plt.plot(amp_i / 1.e-9, '.-')
        mean = amp_i.mean() / 1e-9
        err = amp_i.std() / 1e-9
        plt.xlabel('measurements index', size=10)
        plt.ylabel('Flat Surface amplitude [nm] rms', size=10)
        plt.hlines(mean, 0, 42, colors='r', linestyle='-',
                   linewidth=0.7, label='Mean')
        plt.hlines(mean + err, 0, 42, colors='r',
                   linestyle='--', linewidth=0.7)
        plt.hlines(mean - err, 0, 42, colors='r',
                   linestyle='--', linewidth=0.7)

        wf_mean, wf_sigma = fsa.collapse_normal_flat_maps()
        plt.figure()
        plt.clf()
        plt.imshow(wf_mean / 1e-9)
        plt.colorbar(label='[nm]')
        plt.figure()
        plt.clf()
        plt.imshow(wf_sigma / 1e-9)
        plt.colorbar(label='[nm]')
        # con outlier
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
            plt.plot(vv, mcl._finter[act](vv) / 1.e-6, '-', label='finter')
        plt.xlabel('Command [au]', size=10)
        plt.ylabel('Stroke [$\mu$m]', size=10)
        #plt.title('Calibration curve per actuator', size=15)
        plt.grid()

    def display_acts_ptv_calibrations(self):
        import matplotlib.pyplot as plt
        mcl = self.mcl
        peak = mcl._deflection.max(axis=1)
        valley = mcl._deflection.min(axis=1)
        pos = peak - valley
        dis = Boston140Display()
        plt.figure()
        plt.clf()
        plt.imshow(dis.map(pos) / 1e-6)
        plt.colorbar(label='peak to valley stroke [$\mu$m]')

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

    def show_crosstalk_next_to_act(self, act=76, pos_str='1500nm'):
        fname = 'prova/misure_crosstalk/act76_pos' + pos_str + '.fits'
        cta = CrossTalkAnalyzer(fname)

        cta.show_crosstalk_along_x_axis()
        cta.show_crosstalk_along_y_axis()
        cta.measure_crosstalk()

    def show_act_coupling_next_to(self, act=76):
        import matplotlib.pyplot as plt
        fname = 'prova/misure_coupling/cm_centralact%d' % act + '_3x3.fits'
        cm = CouplingMeasurer(act - 12 + 1, 3)
        cm.pos, cm.act_list, cm.wfs_poke, cm.wf_pos, cm.wf_neg, cm.act_pix_coord = CouplingMeasurer.load(
            fname)
        dis = Boston140Display()
        pos_poke = cm.get_poke_act_stoke()
        pos_ptv = cm.get_actuators_ptv()
        plt.figure()
        plt.clf()
        plt.imshow(dis.map(pos_poke) / 1e-9)
        plt.colorbar()
        plt.figure()
        plt.clf()
        plt.imshow(dis.map(pos_ptv) / 1e-9)
        plt.colorbar()

    def show_flattening_results(self, j_idx):
        import matplotlib.pyplot as plt
        fname = 'prova/misure_con_tappo/misure_spianamento/dati_spianamento_per_plot.fits'
        flatten_data, pos_list, thres_list, n_points, n_flatten = MemsFlatReshaper.load_flatten_data(
            fname)
        acts_per_thres = np.array([140, 104, 80, 76, 64])
        col = ['k', 'g', 'r', 'b', 'c']
        lstyle = ['.', '+', '*']
        plt.figure()
        plt.clf()
        for i, thres in enumerate(thres_list):
            for j, pos in enumerate(pos_list):
                plt.plot(flatten_data[i, j, 0, :] / 1e-9, '-' +
                         col[i] + lstyle[j], lw=0.2, label='thres = %g' % thres + ' start_wf = %g nm' % pos)
        plt.legend(loc='best')
        plt.xlabel('flattening iterations', size=10)
        plt.ylabel('Surface amplitude rms [nm]', size=10)

        plt.figure()
        plt.clf()
        data = flatten_data.mean(axis=2)
        yerr = flatten_data.std(axis=2)
        for i, thres in enumerate(thres_list):
            # j=2 #rand pos 1e-6
            j = j_idx  # flat
            plt.plot(data[i, j, :] / 1e-9, '-' +
                     col[i] + '.', lw=0.2, label='%d actuators' % acts_per_thres[i], markersize=0.8)
            plt.errorbar(range(0, 11), data[i, j, :] / 1e-9, yerr[i, j, :] / 1e-9,
                         fmt='.', color=plt.gca().lines[-1].get_color())

        plt.legend(loc='best')
        plt.xlabel('flattening iterations', size=10)
        plt.ylabel('Surface amplitude rms [nm]', size=10)
        final_ampls = flatten_data.mean(axis=2)[:, :, 2:]
        print('convergence amplitude for wfstart_idx=%d:' % j_idx)
        print(final_ampls.mean(axis=2)[:, j])
        print(final_ampls.std(axis=2)[:, j])
