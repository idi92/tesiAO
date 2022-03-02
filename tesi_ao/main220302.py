from tesi_ao import sandbox
import matplotlib.pyplot as plt
import numpy as np

#_______external function_______


def execute(fname, Nscan, act):

    act_list = [act]

    wyko, bmc = sandbox.create_devices()

    cplm = sandbox.CommandToPositionLinearizationMeasurer(wyko, bmc)

    cplm.NUMBER_STEPS_VOLTAGE_SCAN = Nscan

    cplm.execute_command_scan(act_list)

    cplm.save_results(fname)

    cpla = sandbox.CommandToPositionLinearizationAnalyzer(fname)

    mcl = cpla.compute_linearization()
    plt.ion()
    plt.title('act#63: interpolation functions for several scans')
    plt.xlabel('Commands [au]')
    plt.ylabel('Deflection [m]')

    plt.plot(mcl._cmd_vector[0], mcl._deflection[0], 'o',
             label='%d scans' % cplm.NUMBER_STEPS_VOLTAGE_SCAN)

    Npt = 1000
    f_int = mcl._finter[0]
    span = np.linspace(
        min(mcl._deflection[0]), max(mcl._deflection[0]), Npt)
    plt.plot(f_int(span), span, '-', color=plt.gca().lines[-1].get_color())
    plt.legend(loc='best')
    return f_int, mcl

#_______main_______


def plot_interpolation_difference(under_score_name, act):

    act_list = [act]
    fpath = 'prova/act63/main/cplm'
    ffmt = '.fits'
    wyko, bmc = sandbox.create_devices()

    num_scans_list = np.array([11, 22, 33, 44, 55, 66])

    f_intepol_list = [0, 0, 0, 0, 0, 0]

    mcl_list = [0, 0, 0, 0, 0, 0]

    plt.figure(1001)
    plt.clf()
    plt.grid()

    for index, Nscan in enumerate(num_scans_list):
        print('\n%d voltage scans:' % Nscan)
        fname = fpath + '%d' % Nscan + under_score_name + ffmt
        f_intepol_list[index], mcl_list[index] = execute(fname, Nscan, act)

    plt.figure(1002)
    plt.clf()
    Npt = 1024

    min_container = np.zeros(len(num_scans_list))
    max_container = np.zeros(len(num_scans_list))

    # searching common deflection range
    for i in np.arange(len(num_scans_list)):
        min_container[i] = min(mcl_list[i]._deflection[0])
        max_container[i] = max(mcl_list[i]._deflection[0])

    deflection_span = np.linspace(
        max(min_container), min(max_container), Npt)
    # interpolation function reference: 22scans
    f_2 = f_intepol_list[1]

    for i, Nscan in enumerate(num_scans_list):
        f_i = f_intepol_list[i]
        plt.plot(deflection_span, f_i(deflection_span) - f_2(deflection_span),
                 '.-', label='%d scans' % Nscan)

    plt.legend(loc='best')
    plt.grid()
    plt.ylabel('Command Difference [au]')
    plt.xlabel('Deflection [m]')
    plt.title('act#%d: cubic interpolation error w-r-t 22scans' % act)
