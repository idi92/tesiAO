import numpy as np
from tesi_ao.mems_interpolation_error import LinearityResponseAnalyzer, InterpolationErrorAnalyzer
from tesi_ao.mems_command_linearization import MemsCommandLinearization
from tesi_ao.mems_zernike_mode_analyzer import MemsModeMeasurer,\
    MemsAmplitudeLinearityAnalizer
from tesi_ao.mems_zonal_influence_functions_measurer import ZonalInfluenceFunctionMeasurer
from tesi_ao.main220316 import create_devices
from astropy.io import fits


def main(act=63, n_points=10, Ntimes=100):
    '''
    qualcosa non va: il 63 non riesce a raggiungere
    il minimo stoke previsto dalla sua calibrazione
    quella nel mcl_all_def.fits
    '''
    fname = 'prova/misure_ripetute/mcl_all_def.fits'
    mcl = MemsCommandLinearization.load(fname)
    mcl.clipping_vector = np.zeros(140)
    b = mcl._calibrated_position[act].min()
    a = mcl._calibrated_position[act].max()
    exp_pos = np.linspace(a - 20e-9, b + 20e-9, n_points)
    iea = InterpolationErrorAnalyzer(act, None, n_points)
    iea.execute_multiple_linear_measure(mcl, Ntimes, exp_pos)
    n_scan = len(mcl._cmd_vector[act])
    fname_fit = 'prova/mems_interpolation_error/dati_per_fit_act%d' % act + \
        '_nscan%d' % n_scan + 'times%d' % Ntimes + '_220630c.fits'
    iea.save_linear_results(n_scan, fname_fit)


def do_linear_fitting(fname_fit):
    '''
    qualcosa non va: il 63 non riesce a raggiungere
    il minimo stoke previsto dalla sua calibrazione
    quella nel mcl_all_def.fits
    prima raggiungeva -800nm ora arriva a -700nm
    '''
    if fname_fit is None:
        fname_fit = 'prova/mems_interpolation_error/dati_per_fit_act63_nscan20times3_220630c.fits'

    lra = LinearityResponseAnalyzer(fname_fit)
    lra.show_meas_vs_exp()
    par, cov, chisq = lra.execute_fit()
    lra.show_meas_vs_exp()
    lra.show_meas_vs_fit()
    print('fit parameters:')
    print(par)
    print('sigma:')
    print(np.sqrt(np.diagonal(cov)))
    print('Chi^2 = %g' % chisq)
    red = chisq / (len(lra.expected_pos) - 2)
    print('chi^2/dof = %g' % red)


def try_with_new_calibration_curve(act=63, n_points=10, Ntimes=10):
    '''
    provo quindi a ripetere la calibrazione per il 63
    '''
    iea = InterpolationErrorAnalyzer(act, [20], n_points)
    iea.fversion = '_220630'
    iea.execute_multiple_scans()
    iea.compute_multiple_scans_interpolation()
    mcl_list = iea.load_multiple_scans_interpolation()
    iea._plot_interpolation_function(mcl_list[0])
    mcl_list = iea.load_multiple_scans_interpolation()
    iea.execute_multiple_linear_measure(mcl_list[0], Ntimes, None)
    iea.save_linear_results(20, 'prova/prova_act63.fits')


def try_linear_fit():
    lra = LinearityResponseAnalyzer('prova/prova_act63.fits')
    lra.show_meas_vs_exp()
    lra.show_meas_vs_fit()
    par, cov, chi2 = lra.execute_fit()
    return par, cov, chi2


# def check_min_max_act_stoke():
#     mmm = MemsModeMeasurer()
#     mmm.create_mask()
#     mmm.create_reconstructor(0.15)
#     act_list = mmm.selected_actuators
#     fname = 'prova/misure_ripetute/mcl_all_def.fits'
#     mcl = MemsCommandLinearization.load(fname)


def example_zifs_measure(act_list=[76], push_pull=500e-9):
    wyko, bmc = create_devices()
    zifm = ZonalInfluenceFunctionMeasurer(wyko, bmc)
    zifm.execute_ifs_measure(push_pull, act_list)
    fname = 'prova/example_zifm_meas_pushpull500nm_act%d' % act_list[0] + '.fits'
    zifm._save_meas(fname)
    zifm.compute_zonal_ifs()
    zifm.save_ifs('prova/example_ifs76_pushpull500nm.fits')


def check_min_max_stroke220701():
    fname = 'prova/misure_ripetute/mcl_all_def.fits'
    mcl = MemsCommandLinearization.load(fname)
    mmm = MemsModeMeasurer()
    mmm.create_mask()
    mmm.create_reconstructor(0.15)
    act_list = mmm.selected_actuators
    n_act = len(act_list)
    max_meas_deflection = np.zeros(n_act)
    min_meas_deflection = np.zeros(n_act)
    wyko, bmc = create_devices()
    cmd0 = np.zeros(140)
    cmd = np.zeros(140)
    bmc.set_shape(cmd0)

    def max_wavefront(wf, wf_ref):
        coord_max = np.argwhere(np.abs(wf_ref) == np.max(np.abs(wf_ref)))[0]
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

    for idx, act in enumerate(act_list):
        bmc.set_shape(cmd0)
        wf_flat = wyko.wavefront()
        cmd[int(act)] = mcl._cmd_vector[int(act), 0]
        bmc.set_shape(cmd)
        wf_max = wyko.wavefront()
        dd = wf_max - wf_flat
        wf_max = dd - np.ma.median(dd)
        max_meas_deflection[idx] = max_wavefront(wf=wf_max, wf_ref=wf_max)

        bmc.set_shape(cmd0)
        wf_flat = wyko.wavefront()
        cmd[int(act)] = mcl._cmd_vector[int(act), -1]
        bmc.set_shape(cmd)
        wf_min = wyko.wavefront()
        dd = wf_min - wf_flat
        wf_min = dd - np.ma.median(dd)
        min_meas_deflection[idx] = max_wavefront(wf=wf_min, wf_ref=wf_max)
        cmd = np.zeros(140)

        peak140 = np.zeros(140)
        valley140 = np.zeros(140)
        peak140[act_list] = max_meas_deflection
        valley140[act_list] = min_meas_deflection
        fits.writeto('prova/verifica_act_scalibrati.fits', peak140)
        fits.append('prova/verifica_act_scalibrati.fits', valley140)
        fits.append('prova/verifica_act_scalibrati.fits', act_list)

    return min_meas_deflection, max_meas_deflection, act_list


def measure_some_modes_with_new_calib_220701(j_list):
    fname = 'prova/trm_mcl_all_mod_220701.fits'
    #mcl = MemsCommandLinearization.load(fname)
    mmm = MemsModeMeasurer(ifs_fname=None, mcl_fname=fname)
    mmm.create_mask()
    mmm.create_reconstructor(0.15)
    mmm.create_zernike2zonal()
    #j_list = np.array([2, 3, 4, 11])
    # j_list = np.arange(2, 51)
    for j in j_list:

        mmm.execute_amplitude_scan_for_zernike(j)
        fname = 'prova/mems_mode_measurements/220701a/z%d_act80.fits' % j
        mmm.save_results(fname)


def analyze_and_comapre_with_male220701(j_index, amp_idx):
    import matplotlib.pylab as plt
    fname_male_a = 'prova/mems_mode_measurements/220629/z%d_act80.fits' % j_index
    male_a = MemsAmplitudeLinearityAnalizer(fname_male_a)
    fname_male_b = 'prova/mems_mode_measurements/220701a/z%d_act80.fits' % j_index
    male_b = MemsAmplitudeLinearityAnalizer(fname_male_b)
    print('when mems clips:')
    print(male_a.get_clipped_modes_index_list())
    print(male_a.get_clipped_modes_index_list())
    # TODO assert to avoid clipped modes
    plt.figure()
    plt.clf()
    plt.imshow(male_a.wfs[amp_idx])
    plt.title('MALE_A: amp_meas %g m' %
              male_a.get_measured_amplitudes()[amp_idx])
    plt.colorbar()
    plt.figure()
    plt.clf()
    plt.imshow(male_b.wfs[amp_idx])
    plt.title('MALE_B: amp_meas %g m' %
              male_b.get_measured_amplitudes()[amp_idx])
    plt.colorbar()

    plt.figure()
    plt.clf()
    plt.imshow(male_a.wfs[amp_idx] - male_b.wfs[amp_idx])
    plt.colorbar()
    plt.title('MALE_A-MALE_B')
    a_exp = male_a.get_expected_amplitudes()[amp_idx]
    fit_err_map_a = male_a.wfs[amp_idx] - a_exp * male_a.z_mode
    fitting_err_a = fit_err_map_a.std()
    print('A')
    print(fitting_err_a)
    fit_err_map_b = male_b.wfs[amp_idx] - a_exp * male_b.z_mode
    fitting_err_b = fit_err_map_b.std()
    print('B')
    print(fitting_err_b)
    plt.figure()
    plt.clf()
    plt.imshow(fit_err_map_a)
    plt.colorbar()
    plt.title('fitting error map A')
    plt.figure()
    plt.clf()
    plt.imshow(fit_err_map_b)
    plt.colorbar()
    plt.title('fitting error map B')
    return male_a, male_b
