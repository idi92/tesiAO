from tesi_ao import sandbox, package_data
import matplotlib.pyplot as plt
import numpy as np


def plot_calibration_reproducibility():
    '''
    mcl.fits, mcl1.fits, mcl2.fits sono 3 misure di calibrazione ripetute in rapida sequenza
    '''

    fname0 = package_data.file_name_mcl('mcl0')
    fname1 = package_data.file_name_mcl('mcl1')
    fname2 = package_data.file_name_mcl('mcl2')
    mcl0 = sandbox.MemsCommandLinearization.load(fname0)
    mcl1 = sandbox.MemsCommandLinearization.load(fname1)
    mcl2 = sandbox.MemsCommandLinearization.load(fname2)

    plt.plot(mcl0._cmd_vector[3], mcl1._deflection[3] -
             mcl0._deflection[3], '.-', label='meas1')
    plt.plot(mcl0._cmd_vector[3], mcl2._deflection[3] -
             mcl0._deflection[3], '.-', label='meas2')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Command [au]')
    plt.ylabel('Deflection error wrt meas0 [au]')


def main_calibrate_all_actuators():
    wyko, bmc = sandbox.create_devices()
    mcl, cplm, cpla = sandbox.main_calibration(
        wyko, bmc, mcl_fname='/tmp/mcl_all.fits', scan_fname='/tmp/cpl_all.fits')
    return mcl, cplm, cpla


def max_wavefront(wf):
    coord_max = np.argwhere(
        np.abs(wf) == np.max(np.abs(wf)))[0]
    return wf[coord_max[0], coord_max[1]], coord_max
