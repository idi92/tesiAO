from tesi_ao import mems_zernike_mode_analyzer


def main(j_start=2):
    mmm = mems_zernike_mode_analyzer.MemsModeMeasurer()
    mmm.create_mask()
    mmm.create_reconstructor(0.15)
    mmm.create_zernike2zonal()

    for i in range(j_start, 51):

        mmm.execute_amplitude_scan_for_zernike(i)
        fname = 'prova/mems_mode_measurements/220629/z%d_act80.fits' % i
        mmm.save_results(fname)
