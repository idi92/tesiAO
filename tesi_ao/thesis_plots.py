import numpy as np
from tesi_ao import main220316
from tesi_ao.mems_command_linearization import MemsCommandLinearization
from tesi_ao.mems_interpolation_error import InterpolationErrorAnalyzer, LinearityResponseAnalyzer
from tesi_ao.mems_cross_talk import CrossTalkAnalyzer
from tesi_ao.mems_flat_stability import FlatStabilityAnalyzer
from tesi_ao.mems_flat_reshaper import MemsFlatReshaper
from tesi_ao.mems_max_stoke_coupling import CouplingMeasurer
from tesi_ao.mems_display import Boston140Display
from tesi_ao.mems_reconstructor import MemsZonalReconstructor
from tesi_ao.mems_zonal_influence_functions_measurer import ZonalInfluenceFunctionMeasurer
from pywt._thresholding import threshold
from tesi_ao.mems_command_to_position_linearization_analyzer import CommandToPositionLinearizationAnalyzer
from numpy.lib.tests.test_format import dtype


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

    def display_acts_ptv_calibrations(self, mcl=None):
        import matplotlib.pyplot as plt
        if mcl is None:
            mcl = self.mcl
        peak = mcl._deflection.max(axis=1)
        valley = mcl._deflection.min(axis=1)
        pos = peak - valley
        dis = Boston140Display()
        plt.figure()
        plt.clf()
        plt.imshow(dis.map(pos) / 1e-6)
        plt.colorbar(label='peak to valley stroke [$\mu$m]')
        self._add_acts_labels_to_display()

    def display_acts_ptv_difference_between_calibration(self, mcl_odd):
        import matplotlib.pyplot as plt
        mcl_ref = self.mcl
        peak = mcl_ref._deflection.max(axis=1)
        valley = mcl_ref._deflection.min(axis=1)
        pos_ref = peak - valley
        peak = mcl_odd._deflection.max(axis=1)
        valley = mcl_odd._deflection.min(axis=1)
        pos_odd = peak - valley
        dpos = pos_odd - pos_ref
        dis = Boston140Display()
        plt.figure()
        plt.clf()
        plt.imshow(dis.map(dpos) / 1e-9)
        plt.colorbar(label='peak to valley stroke difference [nm]')

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
        print(cta.measure_crosstalk())

    def show_act_coupling_next_to(self, act=76):
        import matplotlib.pyplot as plt
        fname = 'prova/misure_coupling/cm_centralact%d' % act + '_3x3.fits'
        cm = CouplingMeasurer(act - 12 + 1, 3)
        cm.pos, cm.act_list, cm.wfs_poke, cm.wf_pos, cm.wf_neg, cm.act_pix_coord = CouplingMeasurer.load(
            fname)
        print(cm.pos)
        dis = Boston140Display()
        pos_poke = cm.get_poke_act_stoke()
        pos_ptv = cm.get_actuators_ptv()
        plt.figure()
        plt.clf()
        plt.imshow(dis.map(pos_poke) / 1e-9, cmap='jet',
                   vmin=pos_poke.min() / 1e-9, vmax=pos_poke.max() / 1e-9)
        plt.colorbar(label='[$nm$]')
        self._add_acts_labels_to_display()
        plt.figure()
        plt.clf()
        plt.imshow(dis.map(pos_ptv) / 1e-6, cmap='jet',
                   vmin=pos_ptv.min() / 1e-9, vmax=pos_ptv.max() / 1e-6)
        plt.colorbar(label='[$\mu m$]')
        self._add_acts_labels_to_display()
        return cm.wf_pos, cm.wf_neg

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
        print('starting amplitude for wfstart_idx=%d:' % j_idx)
        print(data[:, j_idx, 0] / 1e-9)
        print(yerr[:, j_idx, 0] / 1e-9)
        # fare media con errore la somma in quadratura!!! non std
        print('convergence amplitude for wfstart_idx=%d:' % j_idx)
        print('All in nm!')
        for i, thres in enumerate(thres_list):
            weights = 1 / (yerr[i, j, 2:]**2)
            a_i = data[i, j, 2:]
            a_mean = (a_i * weights).sum() / weights.sum()
            err_a_mean = 1 / np.sqrt(weights.sum())
            print('for threshold %g' % thres)
            print(a_mean / 1e-9)
            print(err_a_mean / 1e-9)

        # print('convergence amplitude for wfstart_idx=%d:' % j_idx)
        # print(final_ampls.mean(axis=2)[:, j] / 1e-9)
        # print(final_ampls.std(axis=2)[:, j] / 1e-9)
        # print('All in nm!')

    def show_examples_of_flatten_wfs(self):
        import matplotlib.pyplot as plt
        mfr = MemsFlatReshaper()
        mfr.load_acquired_measures_4plot()
        wfs = mfr.wfs_meas

        def do_plot(wf_input, wf_output):
            print('a_in = %g' % wf_input.std())
            print('a_out = %g' % wf_output.std())
            plt.figure()
            plt.clf()
            plt.imshow(wf_input / 1e-6, cmap='jet',
                       vmin=wf_input.min() / 1e-6, vmax=wf_input.max() / 1e-6)
            plt.colorbar(label='[$\mu$m]')
            plt.figure()
            plt.clf()
            plt.imshow(wf_output / 1e-9, cmap='jet',
                       vmin=wf_output.min() / 1e-9, vmax=wf_output.max() / 1e-9)
            plt.colorbar(label='[nm]')

        print('input surface (1000nm rand):')
        print('selected 140 acts:')
        wf_input = wfs[0, -1, 6, 0]
        wf_output = wfs[0, -1, 6, 5]
        do_plot(wf_input, wf_output)
        print('selected 80 acts:')
        wf_input = wfs[2, -1, 7, 0]
        wf_output = wfs[2, -1, 7, 4]
        do_plot(wf_input, wf_output)
        print('selected 64 acts:')
        wf_input = wfs[4, -1, 0, 0]
        wf_output = wfs[4, -1, 0, 9]
        do_plot(wf_input, wf_output)

        print('output surface (input flat_cmd):')
        print('selected 80 acts:')
        wf_input = wfs[2, 0, 0, 0]
        wf_output = wfs[2, 0, 0, -1]
        do_plot(wf_input, wf_output)

        def do_3Dplot(wf_input, wf_output):
            from mpl_toolkits.mplot3d import Axes3D
            Z1 = wf_input / 1e-6
            frame_shape = wf_input.shape
            X = range(frame_shape[1])
            Y = range(frame_shape[0])
            X, Y = np.meshgrid(X, Y)
            hf1 = plt.figure()
            ha1 = hf1.add_subplot(111, projection='3d')
            map1 = ha1.plot_surface(
                X, Y, Z1, cmap='jet', vmin=Z1.min(), vmax=Z1.max())
            hf1.colorbar(map1)
            Z2 = wf_output / 1e-9
            hf2 = plt.figure()
            ha2 = hf2.add_subplot(111, projection='3d')
            map2 = ha2.plot_surface(
                X, Y, Z2, cmap='jet', vmin=Z2.min(), vmax=Z2.max())
            hf2.colorbar(map2)
        # selected 80 acts:
        wf_input = wfs[2, -1, 7, 0]
        wf_output = wfs[2, -1, 7, 4]
        do_3Dplot(wf_input, wf_output)

    def display_selected_actuators(self, visibility_threshold=0.15):
        import matplotlib.pyplot as plt
        mfr = MemsFlatReshaper()
        mfr.create_mask(radius=120, center=(231, 306))
        mfr.create_reconstructor(set_thresh=visibility_threshold)
        actuators = np.zeros(mfr.number_of_actuators)
        selected_acts = mfr.selected_actuators
        actuators[selected_acts] = 1
        n_of_selected_acts = len(selected_acts)
        dis = Boston140Display()
        plt.figure()
        plt.clf()
        plt.imshow(dis.map(actuators))
        print('visibility threshold set to:%g' % visibility_threshold)
        print('selected acts:%d' % n_of_selected_acts)

    def show_ifs_visibility_on_pupil(self, visibility_threshold=0.15):
        import matplotlib.pyplot as plt
        mfr = MemsFlatReshaper()
        mfr.create_mask(radius=120, center=(231, 306))
        mfr.create_reconstructor(set_thresh=visibility_threshold)
        mfr._mzr._show_actuators_visibility()
        dis = Boston140Display()
        plt.figure()
        plt.clf()
        plt.imshow(dis.map(mfr._mzr._rms_wf) / 1e-9, cmap='jet')
        plt.colorbar(label='$\sigma^{IF}_i$\t[nm]')
        self._add_acts_labels_to_display()

    def show_normalized_zifs_for_act(self, act=76):
        import matplotlib.pylab as plt
        mfr = MemsFlatReshaper()
        mfr.create_mask()
        mfr.create_reconstructor()
        ifs = mfr._mzr._ifs
        norm_ifs = mfr._mzr._normalized_ifs
        plt.figure()
        plt.clf()
        plt.imshow(ifs[act] / 1e-9, cmap='jet')
        plt.colorbar(label='[nm]')
        # plt.figure()
        # plt.clf()
        # plt.imshow(norm_ifs[act])
        # plt.colorbar(label='normalized')
        coord_max = np.argwhere(
            np.abs(norm_ifs[act]) == np.max(np.abs(norm_ifs[act])))[0]
        y, x = coord_max[0], coord_max[1]
        plt.figure()
        plt.clf()
        plt.imshow(norm_ifs[act, y - 50:y + 50, x - 50:x + 50], cmap='jet')
        plt.colorbar(label='normalized')
        plt.figure()
        plt.clf()
        plt.imshow(ifs[act, y - 50:y + 50, x - 50:x + 50] / 1e-9, cmap='jet')
        plt.colorbar(label='[nm]')

        plt.figure()
        plt.clf()
        plt.plot(norm_ifs[act, y, x - 60:x + 60])
        plt.xlabel('roi pixels along x axis', size=10)

        plt.figure()
        plt.clf()
        plt.plot(norm_ifs[act, y - 60:y + 60, x])
        plt.xlabel('roi pixels along y axis', size=10)
        print(y, x)

    def show_example_zifmeas_process(self):
        import matplotlib.pylab as plt
        fname = 'prova/example_zifm_meas_pushpull500nm_act76.fits'
        stroke, act_list, wfs_pos, wfs_neg, wfs_zero = ZonalInfluenceFunctionMeasurer._load_meas(
            fname)
        act = act_list[0]
        wfp = wfs_pos[act] - wfs_zero[act]
        wfn = wfs_neg[act] - wfs_zero[act]
        coord_max = np.argwhere(
            np.abs(wfp) == np.max(np.abs(wfp)))[0]
        yp, xp = coord_max[0], coord_max[1]
        coord_max = np.argwhere(
            np.abs(wfn) == np.max(np.abs(wfn)))[0]
        yn, xn = coord_max[0], coord_max[1]

        plt.figure()
        plt.imshow(wfp[yp - 50: yp + 50, xp - 50: xp + 50] /
                   1e-9, vmax=500, cmap='jet', aspect='auto')
        plt.colorbar(label='[nm]')
        plt.figure()
        plt.imshow(wfn[yn - 50: yn + 50, xn - 50: xn + 50] /
                   1e-9, vmin=-500, cmap='jet', aspect='auto')
        plt.colorbar(label='[nm]')
        dd = wfp - wfn
        wf_ifs = (dd - np.ma.median(dd))
        coord_max = np.argwhere(
            np.abs(wfn) == np.max(np.abs(wfn)))[0]
        y, x = coord_max[0], coord_max[1]
        plt.figure()
        plt.imshow(wf_ifs[y - 50: y + 50, x -
                          50: x + 50] / 1e-9, cmap='jet', aspect='auto')
        plt.colorbar(label='[nm]')
        plt.figure()
        plt.imshow(0.5 * wf_ifs[y - 50: y + 50, x -
                                50: x + 50] / stroke, cmap='jet', aspect='auto')
        plt.colorbar()

    def show_example_3D_IFS(self):
        import matplotlib.pylab as plt
        from mpl_toolkits.mplot3d import Axes3D
        fname = 'prova/example_zifm_meas_pushpull500nm_act76.fits'
        stroke, act_list, wfs_pos, wfs_neg, wfs_zero = ZonalInfluenceFunctionMeasurer._load_meas(
            fname)
        act = act_list[0]
        wfp = wfs_pos[act] - wfs_zero[act]
        wfn = wfs_neg[act] - wfs_zero[act]
        dd = wfp - wfn
        wf_ifs = (dd - np.ma.median(dd))
        coord_max = np.argwhere(
            np.abs(wfn) == np.max(np.abs(wfn)))[0]
        y, x = coord_max[0], coord_max[1]
        ifs = 0.5 * wf_ifs[y - 50: y + 50, x - 50: x + 50] / stroke
        Z = ifs
        n = Z.shape
        print(n)
        X = range(n[0])
        Y = range(n[1])
        X, Y = np.meshgrid(X, Y)
        hf = plt.figure()
        ha = hf.add_subplot(111, projection='3d')
        map = ha.plot_surface(X, Y, Z, cmap='jet', vmin=Z.min(), vmax=Z.max())
        hf.colorbar(map)
        # ha.set_axis_off()

    def show_deformed_calibration_curves(self):
        import matplotlib.pyplot as plt
        mcl = MemsCommandLinearization.load(
            'prova/misure_ripetute/mcl_all_def.fits')
        mcl_bozzi = MemsCommandLinearization.load(
            'prova/misure_con_tappo/trm_mcl_all_mod.fits')
        plt.figure()
        plt.clf()
        plt.plot(
            mcl._calibrated_cmd[15], mcl._calibrated_position[15] / 1e-6, 'r--', lw=0.9)
        plt.plot(
            mcl_bozzi._calibrated_cmd[15], mcl_bozzi._calibrated_position[15] / 1e-6, 'r-', label='15', lw=0.9)
        plt.plot(
            mcl._calibrated_cmd[15], mcl._calibrated_position[113] / 1e-6, 'b--', lw=0.9)
        plt.plot(
            mcl_bozzi._calibrated_cmd[15], mcl_bozzi._calibrated_position[113] / 1e-6, 'b-', label='113', lw=0.9)
        plt.xlabel('Command [au]', size=10)
        plt.ylabel('Surface deflection [$\mu$m]', size=10)
        ptv = mcl._deflection[:, 0] - mcl._deflection[:, -1]
        ptv_bozzi = mcl_bozzi._deflection[:, 0] - mcl_bozzi._deflection[:, -1]
        print('Act 15:')
        print('ptv_mcl:%g' % ptv[15])
        print('ptv_mcl:%g' % ptv_bozzi[15])
        print('Act 113:')
        print('ptv_mcl:%g' % ptv[113])
        print('ptv_mcl:%g' % ptv_bozzi[113])
        plt.legend(loc='best')
        plt.figure()
        plt.clf()
        plt.plot(ptv / 1e-6, '.')
        plt.plot(ptv_bozzi / 1e-6, '.-')

    def show_SVD_eigenvalues(self, vis_threshold=0.15):
        import matplotlib.pyplot as plt
        mfr = MemsFlatReshaper()
        mfr.create_mask(radius=120, center=(231, 306))
        mfr.create_reconstructor(set_thresh=vis_threshold)
        mfr._mzr._get_svd()
        Nact = mfr.number_of_selected_actuators
        index = np.arange(1, Nact + 1)
        singular_values = mfr._mzr.s
        plt.figure()
        plt.plot(index, singular_values, '.-')
        plt.ylabel('Eigenvalues\t$\lambda_i$', size=10)
        plt.xlabel('mode index', size=10)
        plt.tight_layout()

    def show_SVD_eigenvectors(self, vis_threshold=0.15, eigen_index=0):
        import matplotlib.pyplot as plt
        mfr = MemsFlatReshaper()
        mfr.create_mask(radius=120, center=(231, 306))
        mfr.create_reconstructor(set_thresh=vis_threshold)
        mfr._mzr._get_svd()
        wf = np.zeros((486, 640))
        wf[mfr.cmask == False] = np.dot(
            mfr.im, mfr._mzr.vh.T[:, eigen_index])
        eigenmap = np.ma.array(wf, mask=mfr.cmask)
        plt.figure()
        plt.imshow(eigenmap)
        plt.colorbar()

    def show_SVD_test(self, vis_threshold=0.15):
        import matplotlib.pyplot as plt
        mfr = MemsFlatReshaper()
        mfr.create_mask(radius=120, center=(231, 306))
        mfr.create_reconstructor(set_thresh=vis_threshold)
        Nact = mfr.number_of_selected_actuators
        mfr._mzr._get_svd()

        fig, axs = plt.subplots(5, 10)

        eigen_index = 0

        wffirst = np.zeros((486, 640))
        wffirst[mfr.cmask == False] = np.dot(
            mfr.im, mfr._mzr.vh.T[:, eigen_index])
        first_eigenmap = np.ma.array(wffirst, mask=mfr.cmask)
        min_val = first_eigenmap.min()
        max_val = first_eigenmap.max()

        for ax in axs.flat:
            wf = np.zeros((486, 640))
            wf[mfr.cmask == False] = np.dot(
                mfr.im, mfr._mzr.vh.T[:, eigen_index])
            eigenmap = np.ma.array(wf, mask=mfr.cmask)
            im = ax.imshow(eigenmap, vmin=min_val, vmax=max_val,
                           cmap='jet', aspect='auto')
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)
            eigen_index += 1

        fig.colorbar(im, ax=axs.ravel().tolist())
        # fig.colorbar(im)

        # for i in range(5):
        #     for j in range(10):
        #         wf = np.zeros((486, 640))
        #         wf[mfr.cmask == False] = np.dot(
        #             mfr.im, mfr._mzr.vh.T[:, eigen_index])
        #         eigenmap = np.ma.array(wf, mask=mfr.cmask)
        #         axs[i, j].imshow(eigenmap, vmin=min_val, vmax=max_val,
        #                          cmap='jet', aspect='auto')
        #         axs[i, j].get_yaxis().set_visible(False)
        #         axs[i, j].get_xaxis().set_visible(False)
        #
        #         #axs[i, j].set_title('$\lambda_{%d}$' % eigen_index)
        #         eigen_index += 1

    def show_old_measurements_visibility(self):
        import matplotlib.pyplot as plt
        cpla_fname = 'prova/all_act/sandbox/cplm_all_fixed.fits'
        mcl_fname = 'prova/all_act/sandbox/mcl_all_fixedpix.fits'
        cpla = CommandToPositionLinearizationAnalyzer(cpla_fname)
        mcl = MemsCommandLinearization.load(mcl_fname)
        mg = main220316.ModeGenerator(cpla, mcl)
        mg.NORM_AT_THIS_CMD = 0
        mg.VISIBLE_AT_THIS_CMD = mg.NORM_AT_THIS_CMD
        mg.compute_reconstructor(mask_obj=None)
        dis = Boston140Display()
        plt.figure()
        plt.clf()
        plt.imshow(dis.map(mg._rms_wf) / 1e-9, vmin=0,
                   vmax=mg._rms_wf.max() / 1e-9, cmap='jet')
        plt.colorbar(label='$\sigma^{max}_i$\t[nm]')
        self._add_acts_labels_to_display()
        # act_text = np.ma.zeros((12, 12), dtype=int)
        # act_text = np.ma.array(data=act_text, mask=act_text)
        # act_text.mask[0, 0] = True
        # act_text.mask[-1, 0] = True
        # act_text.mask[0, -1] = True
        # act_text.mask[-1, -1] = True
        # act_text.data[act_text.mask == False] = np.arange(140)
        # for (i, j), txt_label in np.ndenumerate(dis.map(np.arange(140))):
        #     if (act_text.mask[i, j] == False):
        #         plt.text(j, i, int(txt_label), ha='center', va='center')

    def show_old_detector_mask_example(self):
        import matplotlib.pyplot as plt
        cpla_fname = 'prova/all_act/sandbox/cplm_all_fixed.fits'
        cpla = CommandToPositionLinearizationAnalyzer(cpla_fname)
        wfs = cpla._wfs
        cmd = 0
        plt.figure()
        plt.imshow(wfs[4, cmd] / 1e-9, cmap='jet')
        plt.colorbar(label='$[nm]$')
        plt.figure()
        plt.imshow(wfs[15, cmd] / 1e-9, cmap='jet')
        plt.colorbar(label='$[nm]$')
        plt.figure()
        plt.imshow(wfs[27, cmd] / 1e-9, cmap='jet')
        plt.colorbar(label='$[nm]$')
        plt.figure()
        px = 257
        scale = 1e-6
        plt.plot(wfs[4, cmd, px, :] / scale, '-', label='4')
        plt.plot(wfs[15, cmd, px, :] / scale, '-', label='15')
        plt.plot(wfs[27, cmd, px, :] / scale, '-', label='27')
        plt.plot(wfs[111, cmd, px, :] / scale, '-', label='111')
        plt.plot(wfs[123, cmd, px, :] / scale, '-', label='123')
        plt.plot(wfs[134, cmd, px, :] / scale, '-', label='134')

        Z1 = wfs[4, cmd] / scale
        Z2 = wfs[15, cmd] / scale
        Z3 = wfs[27, cmd] / scale
        Z4 = wfs[63, cmd] / scale
        z_max = Z4.max()
        z_min = Z4.min()
        n = wfs.shape
        X = range(n[-1])
        Y = range(n[-2])
        X, Y = np.meshgrid(X, Y)

        # hf = plt.figure()
        # ha = hf.add_subplot(111, projection='3d')
        # map = ha.plot_surface(X, Y, Z1, cmap='jet',
        #                       vmin=Z1.min(), vmax=Z1.max())
        # hf.colorbar(map)
        #
        # hf = plt.figure()
        # ha = hf.add_subplot(111, projection='3d')
        # map = ha.plot_surface(X, Y, Z2, cmap='jet',
        #                       vmin=Z2.min(), vmax=Z2.max())
        # hf.colorbar(map)

        n = wfs.shape
        hf = plt.figure()
        ha = hf.add_subplot(111, projection='3d')
        ha.plot_surface(X, Y, Z1, cmap='jet',
                        vmin=z_min, vmax=z_max)
        ha.plot_surface(X + 200, Y + 200, Z2, cmap='jet',
                        vmin=z_min, vmax=z_max)
        ha.plot_surface(X + 400, Y + 400, Z3, cmap='jet',
                        vmin=z_min, vmax=z_max)
        map = ha.plot_surface(X + 600, Y + 600, Z4, cmap='jet',
                              vmin=z_min, vmax=z_max)
        hf.colorbar(map, label='Surface deflection\t' + '$[\mu m]$')
        ha.set_axis_off()

    def _add_acts_labels_to_display(self):
        import matplotlib.pyplot as plt
        dis = Boston140Display()
        act_text = np.ma.zeros((12, 12), dtype=int)
        act_text = np.ma.array(data=act_text, mask=act_text)
        act_text.mask[0, 0] = True
        act_text.mask[-1, 0] = True
        act_text.mask[0, -1] = True
        act_text.mask[-1, -1] = True
        act_text.data[act_text.mask == False] = np.arange(140)
        for (i, j), txt_label in np.ndenumerate(dis.map(np.arange(140))):
            if (act_text.mask[i, j] == False):
                plt.text(j, i, int(txt_label), ha='center', va='center')
