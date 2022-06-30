import numpy as np
from tesi_ao.mems_interpolation_error import LinearityResponseAnalyzer, InterpolationErrorAnalyzer
from tesi_ao.mems_command_linearization import MemsCommandLinearization
from tesi_ao.mems_zernike_mode_analyzer import MemsModeMeasurer
from tesi_ao.mems_zonal_influence_functions_measurer import ZonalInfluenceFunctionMeasurer
from tesi_ao.main220316 import create_devices


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
