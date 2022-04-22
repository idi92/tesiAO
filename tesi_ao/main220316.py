
import numpy as np
from plico_interferometer import interferometer
from plico_dm import deformableMirror
from astropy.io import fits
from scipy.interpolate.interpolate import interp1d
from scipy.interpolate._cubic import CubicSpline

from functools import reduce
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve
from numpy import dtype
from arte.types.mask import CircularMask
from arte.utils.zernike_generator import ZernikeGenerator


def create_devices():
    wyko = interferometer('193.206.155.29', 7300)
    bmc = deformableMirror('193.206.155.92', 7000)
    return wyko, bmc


class CommandToPositionLinearizationMeasurer(object):

    NUMBER_WAVEFRONTS_TO_AVERAGE = 1
    NUMBER_STEPS_VOLTAGE_SCAN = 20
    TIME_OUT = 10  # sec

    def __init__(self, interferometer, mems_deformable_mirror):
        self._interf = interferometer
        self._bmc = mems_deformable_mirror
        self._n_acts = self._bmc.get_number_of_actuators()
        self._wfflat = None

    def _get_zero_command_wavefront(self):
        if self._wfflat is None:
            cmd = np.zeros(self._n_acts)
            self._bmc.set_shape(cmd)
            self._wfflat = self._interf.wavefront(
                self.NUMBER_WAVEFRONTS_TO_AVERAGE, timeout_in_sec=self.TIME_OUT)
        return self._wfflat

    def execute_command_scan(self, act_list=None):
        if act_list is None:
            act_list = np.arange(self._n_acts)

        self._actuators_list = np.array(act_list)
        n_acts_to_meas = len(self._actuators_list)

        wfflat = self._get_zero_command_wavefront()

        self._reference_cmds = self._bmc.get_reference_shape()
        self._reference_tag = self._bmc.get_reference_shape_tag()

        self._cmd_vector = np.zeros((n_acts_to_meas,
                                     self.NUMBER_STEPS_VOLTAGE_SCAN))

        self._wfs = np.ma.zeros(
            (n_acts_to_meas, self.NUMBER_STEPS_VOLTAGE_SCAN,
             wfflat.shape[0], wfflat.shape[1]))

        N_pixels = self._wfs.shape[2] * self._wfs.shape[3]
        for act_idx, act in enumerate(self._actuators_list):
            self._cmd_vector[act_idx] = np.linspace(
                0, 1, self.NUMBER_STEPS_VOLTAGE_SCAN) - self._reference_cmds[act]
            for cmd_idx, cmdi in enumerate(self._cmd_vector[act_idx]):
                print("Act:%d - command %g" % (act, cmdi))
                cmd = np.zeros(self._n_acts)
                cmd[act] = cmdi
                self._bmc.set_shape(cmd)
                self._wfs[act_idx, cmd_idx, :,
                          :] = self._get_wavefront_flat_subtracted()
                masked_pixels = self._wfs[act_idx, cmd_idx].mask.sum()
                masked_ratio = masked_pixels / N_pixels
                if masked_ratio > 0.7829:
                    print('Warning: Bad measure acquired for: act%d' %
                          act_idx + ' cmd_idx %d' % cmd_idx)
                    self._avoid_saturated_measures(
                        masked_ratio, act_idx, cmd_idx, N_pixels)

    def _avoid_saturated_measures(self, masked_ratio, act_idx, cmd_idx, N_pixels):

        while masked_ratio > 0.7829:
            self._wfs[act_idx, cmd_idx, :,
                      :] = self._get_wavefront_flat_subtracted()
            masked_pixels = self._wfs[act_idx, cmd_idx].mask.sum()
            masked_ratio = masked_pixels / N_pixels

        print('Repeated measure completed!')

    def _get_wavefront_flat_subtracted(self):
        dd = self._interf.wavefront(
            self.NUMBER_WAVEFRONTS_TO_AVERAGE, timeout_in_sec=self.TIME_OUT) - self._get_zero_command_wavefront()
        return dd - np.ma.median(dd)

    def _reset_flat_wavefront(self):
        self._wfflat = None

    def check_mask_coverage(self, ratio=False):
        masked_pixels = np.array([self._wfs[a, i].mask.sum() for a in range(
            self._wfs.shape[0]) for i in range(self._wfs.shape[1])])
        titlestr = 'Number'
        if(ratio == True):
            masked_pixels = masked_pixels / \
                (self._wfs.shape[2] * self._wfs.shape[3])
            titlestr = 'Fraction'
        plt.figure()
        plt.clf()
        plt.ion()
        plt.plot(masked_pixels)

        plt.ylabel(titlestr + ' of Masked Pixels', size=25)
        plt.xlabel('Measures', size=25)
        plt.title('Number of scans per actuator:%d' %
                  self._wfs.shape[1])

    def add_repeated_measure(self, cplm_to_add, act_list):
        for idx, act in enumerate(act_list):
            self._wfs[act] = cplm_to_add._wfs[idx]

    def save_results(self, fname):
        hdr = fits.Header()
        hdr['REF_TAG'] = self._reference_tag
        hdr['N_AV_FR'] = self.NUMBER_WAVEFRONTS_TO_AVERAGE
        fits.writeto(fname, self._wfs.data, hdr)
        fits.append(fname, self._wfs.mask.astype(int))
        fits.append(fname, self._cmd_vector)
        fits.append(fname, self._actuators_list)
        fits.append(fname, self._reference_cmds)

    @staticmethod
    def load(fname):
        header = fits.getheader(fname)
        hduList = fits.open(fname)
        wfs_data = hduList[0].data
        wfs_mask = hduList[1].data.astype(bool)
        wfs = np.ma.masked_array(data=wfs_data, mask=wfs_mask)
        cmd_vector = hduList[2].data
        actuators_list = hduList[3].data
        reference_commands = hduList[4].data
        return {'wfs': wfs,
                'cmd_vector': cmd_vector,
                'actuators_list': actuators_list,
                'reference_shape': reference_commands,
                'reference_shape_tag': header['REF_TAG']
                }


class CommandToPositionLinearizationAnalyzer(object):

    def __init__(self, scan_fname):
        res = CommandToPositionLinearizationMeasurer.load(scan_fname)
        self._wfs = res['wfs']
        self._cmd_vector = res['cmd_vector']
        self._actuators_list = res['actuators_list']
        self._reference_shape_tag = res['reference_shape_tag']
        self._n_steps_voltage_scan = self._wfs.shape[1]

    # def _max_wavefront(self, act_idx, cmd_index):
    #     wf = self._wfs[act_idx, cmd_index]
    #     coord_max = np.argwhere(np.abs(wf) == np.max(np.abs(wf)))[0]
    #     return wf[coord_max[0], coord_max[1]]

    # def _max_roi_wavefront(self, act_idx, cmd_index):
    #     wf = self._wfs[act_idx, cmd_index]
    #     b, t, l, r = self._get_max_roi(act_idx)
    #     wfroi = wf[b:t, l:r]
    #     print('act%d done!' % act_idx)
    #     coord_max = np.argwhere(
    #         np.abs(wfroi) == np.max(np.abs(wfroi)))[0]
    #     return wfroi[coord_max[0], coord_max[1]]
    #
    def _get_max_roi(self, act):
        roi_size = 50
        wf = self._wfs[act, 0]
        coord_max = self._get_max_pixel(act)
        return coord_max[0] - roi_size, coord_max[0] + roi_size, \
            coord_max[1] - roi_size, coord_max[1] + roi_size

    def _get_max_pixel(self, act):
        wf = self._wfs[act, 2]
        coord_max = np.argwhere(np.abs(wf) == np.max(np.abs(wf)))[0]
        return coord_max[0], coord_max[1]

    def _max_wavefront(self, act, cmd_index):
        wf = self._wfs[act, cmd_index]
        y, x = self._get_max_pixel(act)
        list_to_avarage = []
        # avoid masked data
        for yi in range(y - 1, y + 2):
            for xi in range(x - 1, x + 2):
                if(wf[yi, xi].data != 0.):
                    list_to_avarage.append(wf[yi, xi])
        list_to_avarage = np.array(list_to_avarage)
        # return wf[y, x]
        return np.median(list_to_avarage)

    def _max_vector(self, act_idx):
        print('act%d' % act_idx)
        res = np.zeros(self._n_steps_voltage_scan)
        for i in range(self._n_steps_voltage_scan):
            print('cmd step%d' % i)
            res[i] = self._max_wavefront(act_idx, i)
        return res

    def _compute_maximum_deflection(self):
        self._max_deflection = np.array([
            self._max_vector(act_idx) for act_idx in range(len(self._actuators_list))])

    def compute_linearization(self):
        self._compute_maximum_deflection()

        return MemsCommandLinearization(
            self._actuators_list,
            self._cmd_vector,
            self._max_deflection,
            self._reference_shape_tag)

    def add_repeated_measure(self, cpla_to_add, act_list):
        for idx, act in enumerate(act_list):
            self._wfs[act] = cpla_to_add._wfs[idx]

    def check_mask_coverage(self, ratio=False):
        masked_pixels = np.array([self._wfs[a, i].mask.sum() for a in range(
            self._wfs.shape[0]) for i in range(self._wfs.shape[1])])
        titlestr = 'Number'
        if(ratio == True):
            masked_pixels = masked_pixels / \
                (self._wfs.shape[2] * self._wfs.shape[3])
            titlestr = 'Fraction'
        plt.figure()
        plt.clf()
        plt.ion()
        plt.plot(masked_pixels)

        plt.ylabel(titlestr + ' of Masked Pixels', size=25)
        plt.xlabel('Measures', size=25)
        plt.title('Number of scans per actuator:%d' %
                  self._wfs.shape[1])

    # def _2dgaussian(self, X, amplitude, x0, y0, sigmax, sigmay, offset):
    #     y, x = X
    #     z = np.zeros((len(y), len(x)), dtype='float')
    #     N = amplitude  # *0.5 / (np.pi * sigmax * sigmay)
    #     for xi in np.arange(len(x)):
    #         a = 0.5 * ((xi - x0) / sigmax)**2
    #         for yi in np.arange(len(y)):
    #             b = 0.5 * ((yi - y0) / sigmay)**2
    #
    #             z[yi, xi] = N * np.exp(-(a + b)) + offset
    #     return z.ravel()
    #
    # def _gaussian_fitting(self, act_idx, cmd_index):
    #     wf = self._wfs[act_idx, cmd_index]
    #     wf2 = self._wfs[act_idx, 2]
    #     b, t, l, r = self._get_max_roi(act_idx)
    #     wfroi = wf[b:t, l:r]
    #     wfroi2 = wf2[b:t, l:r]
    #     coord_max = np.argwhere(np.abs(wfroi2) == np.max(np.abs(wfroi2)))[0]
    #     x0 = coord_max[1]
    #     y0 = coord_max[0]
    #     #z = wfroi[wfroi.data != 0.]
    #
    #     #z = wfroi
    #
    #     NvalidX = (wfroi.mask[y0, :] == False).sum()
    #     NvalidY = (wfroi.mask[:, x0] == False).sum()
    #     x = np.arange(NvalidX, dtype='float')
    #     y = np.arange(NvalidY, dtype='float')
    #
    #     Z = []
    #     for yi in range(wfroi.shape[0]):
    #         for xi in range(wfroi.shape[1]):
    #             if(wfroi[yi, xi].data != 0.):
    #                 Z.append(wfroi[yi, xi])
    #
    #     Z = np.array(Z, dtype='float')
    #
    #     Z = wfroi.compressed()
    #
    #     A0 = self._max_wavefront(act_idx, cmd_index)
    #
    #     sigma0 = 25.
    #     sigmax = sigma0
    #     sigmay = sigma0
    #     offset = 0.
    #     starting_values = [A0, x0, y0, sigmax, sigmay, offset]
    #     X = y, x
    #
    #     #err_z = Z.std() * np.ones(len(x) * len(y))
    #
    #     fpar, fcov = curve_fit(self._2dgaussian, X, Z,
    #                            p0=starting_values, absolute_sigma=True)
    #     #err_fpar = np.sqrt(np.diag(fcov))
    #     print('1curve_fit done')
    #     error = (Z - self._2dgaussian(X, *fpar))
    #     starting_values = [fpar[0], fpar[1],
    #                        fpar[2], fpar[3], fpar[4], fpar[5]]
    #     fpar, fcov = curve_fit(
    #         self._2dgaussian, X, Z, p0=starting_values, sigma=error, absolute_sigma=True)
    #     print('2curve_fit done')
    #     return fpar[0]
    #
    # def _compute_gaussian_amplitude_deflection(self):
    #     self._max_deflection = np.zeros(
    #         (self._cmd_vector.shape[0], self._cmd_vector.shape[1]))
    #     for act in range(self._cmd_vector.shape[0]):
    #         for cmd_idx in range(self._cmd_vector.shape[1]):
    #             self._max_deflection[act, cmd_idx] = self._gaussian_fitting(
    #                 act, cmd_idx)
    #
    # def compute_gaussian_linearization(self):
    #     self._compute_gaussian_amplitude_deflection()
    #
    #     return MemsCommandLinearization(
    #         self._actuators_list,
    #         self._cmd_vector,
    #         self._max_deflection,
    #         self._reference_shape_tag)


class MemsCommandLinearization():

    def __init__(self,
                 actuators_list,
                 cmd_vector,
                 deflection,
                 reference_shape_tag):
        self._actuators_list = actuators_list
        self._cmd_vector = cmd_vector
        self._deflection = deflection
        self._reference_shape_tag = reference_shape_tag
        # TODO: dopo aver aggiustato tutto mettere fint giusta
        # self._create_interpolation_test()
        self._create_interpolation()

    def _create_interpolation(self):
        # WARNING: interp 1d suppone che l argomento sia un array
        # di valori monotonicamente crescenti(o comunque li riordina) e
        # le deflessioni non lo sono, per questo motivo in
        # plot_interpolated_functions osservo delle forti oscillazioni
        self._finter = [interp1d(
            self._deflection[i], self._cmd_vector[i], kind='cubic')
            for i in range(self._cmd_vector.shape[0])]

    def _create_interpolation_test(self):
        self._finter = [CubicSpline(self._cmd_vector[i], self._deflection[i], bc_type='natural')
                        for i in range(self._cmd_vector.shape[0])]

    def _get_act_idx(self, act):
        return np.argwhere(self._actuators_list == act)[0][0]

    def p2c(self, act, p):
        idx = self._get_act_idx(act)
        return self._finter[idx](p)

    def solve_p2c(self, act, p):
        '''
        returns required cmd for a given position/deflection
        implemented via scipy.optimize.fsolve
        slows routine?
        '''
        idx = self._get_act_idx(act)

        def func(cmd): return np.abs(p - self._finter[idx](cmd))
        abs_difference = np.abs(p - self._finter[idx](self._cmd_vector[idx]))
        min_abs_difference = abs_difference.min()
        idx_guess = np.where(abs_difference == min_abs_difference)[0][0]
        guess = self._cmd_vector[idx, idx_guess]

        cmd = fsolve(func, x0=guess)
        return cmd[0]

    def linear_p2c(self, act, pos):
        '''
        returns required cmd for a given position/deflection
        implemented via linear interpolation between 2 points
        close to pos and returns the required cmd
        should be faster then solve_p2c
        '''
        # certo se ricreo ogni volta cmd_span e
        # pos_span per ogni act forse sara piu lento
        idx = self._get_act_idx(act)
        cmd_min = self._cmd_vector[idx, 0]
        cmd_max = self._cmd_vector[idx, -1]
        cmd_span = np.linspace(cmd_min, cmd_max, 10000)
        pos_span = self._finter[idx](cmd_span)
        # avro una sensibilita dell ordine di 1.e-4 in tensione,ok
        pos_a = pos_span[pos_span <= pos].max()
        pos_b = pos_span[pos_span >= pos].min()
        idx_cmd_a = np.where(pos_span == pos_a)[0][0]
        idx_cmd_b = np.where(pos_span == pos_b)[0][0]
        x = [pos_b, pos_a]
        y = [cmd_span[idx_cmd_b], cmd_span[idx_cmd_a]]
        f = interp1d(x, y)
        return float(f(pos))

    def save(self, fname):
        hdr = fits.Header()
        hdr['REF_TAG'] = self._reference_shape_tag
        fits.writeto(fname, self._actuators_list, hdr)
        fits.append(fname, self._cmd_vector)
        fits.append(fname, self._deflection)

    @staticmethod
    def load(fname):
        header = fits.getheader(fname)
        hduList = fits.open(fname)
        actuators_list = hduList[0].data
        cmd_vector = hduList[1].data
        deflection = hduList[2].data
        reference_shape_tag = header['REF_TAG']
        return MemsCommandLinearization(
            actuators_list, cmd_vector, deflection, reference_shape_tag)


def main220228():
    mcl = MemsCommandLinearization.load('/tmp/mcl9.fits')
    print('reference shape used when calibrating %s ' %
          mcl._reference_shape_tag)
    actuator_number = 63
    deflection_wrt_reference_shape = 100e-9
    mcl.p2c(actuator_number, deflection_wrt_reference_shape)


def main_calibration(wyko,
                     bmc,
                     mcl_fname='/tmp/mcl0.fits',
                     scan_fname='/tmp/cpl0.fits',
                     act_list=None):
    #wyko, bmc = create_devices()
    cplm = CommandToPositionLinearizationMeasurer(wyko, bmc)

    if act_list is None:
        act_list = np.arange(bmc.get_number_of_actuators())
    cplm.execute_command_scan(act_list)
    cplm.save_results(scan_fname)
    cpla = CommandToPositionLinearizationAnalyzer(scan_fname)
    mcl = cpla.compute_linearization()
    mcl.save(mcl_fname)
    return mcl, cplm, cpla


def plot_interpolated_function(mcl):
    '''
    F_int(pos)=cmd
    '''
    plt.figure()
    plt.clf()
    for idx, act in enumerate(mcl._actuators_list):
        a = np.min(mcl._deflection[act])
        b = np.max(mcl._deflection[act])
        xx = np.linspace(a, b, 1000)
        plt.plot(mcl._finter[act](xx), xx / 1.e-9, '.-')
    plt.xlabel('Command [au]', size=25)
    plt.ylabel('Deflection [nm]', size=25)
    plt.title('Calibration curve per actuator', size=25)
    plt.grid()


def plot_acquired_measures(mcl):
    plt.figure()
    plt.clf()
    for idx, act in enumerate(mcl._actuators_list):
        plt.plot(mcl._cmd_vector[idx], mcl._deflection[idx] / 1.e-9, '.-')
    plt.xlabel('Command [au]', size=25)
    plt.ylabel('Deflection [nm]', size=25)
    plt.title('Acquired Measures per actuator', size=25)
    plt.grid()


def plot_single_curve(mcl, act):
    '''
    F_int(pos)=cmd
    '''
    plt.figure()
    plt.clf()
    a = np.min(mcl._deflection[act])
    b = np.max(mcl._deflection[act])
    xx = np.linspace(a, b, 1000)
    plt.plot(mcl._cmd_vector[act], mcl._deflection[act] /
             1.e-9, 'or', label='sampling points')
    plt.plot(mcl._finter[act](xx), xx / 1.e-9, '-', label='finter')
    plt.title('Calibration Curve: act#%d' % act, size=25)
    plt.xlabel('Commands [au]', size=25)
    plt.ylabel('Deflection [nm]', size=25)
    plt.grid()
    plt.legend(loc='best')


def _plot_pos_vs_cmd(mcl, act):
    '''
    F_int(cmd)=pos
    '''
    plt.figure()
    plt.clf()
    plt.plot(mcl._cmd_vector[act], mcl._deflection[act] /
             1.e-9, 'or', label='sampling points')
    plt.title('act=%d' % act, size=25)
    plt.ylabel('pos[nm]')
    plt.xlabel('cmd[au]')
    plt.grid()
    a = np.min(mcl._cmd_vector[act])
    b = np.max(mcl._cmd_vector[act])
    vv = np.linspace(a, b, 1000)
    plt.plot(vv, mcl._finter[act](vv) / 1.e-9, '-', label='finter')
    plt.legend(loc='best')


def _plot_all_int_funcs(mcl):
    plt.figure()
    plt.clf()
    for idx, act in enumerate(mcl._actuators_list):
        a = np.min(mcl._cmd_vector[act])
        b = np.max(mcl._cmd_vector[act])
        vv = np.linspace(a, b, 1000)
        plt.plot(vv, mcl._finter[act](vv) / 1.e-9, '.-', label='finter')
    plt.xlabel('Command [au]', size=25)
    plt.ylabel('Deflection [nm]', size=25)
    plt.title('Calibration curve per actuator', size=25)
    plt.grid()


class PupilMaskBuilder():

    def __init__(self, wfmask):
        self._wfmask = wfmask  # is the interferometer mask!

    def get_circular_mask(self, radius, center):
        mask = CircularMask(self._wfmask.shape,
                            maskRadius=radius, maskCenter=center)
        return mask  # .mask()
    # TODO: provare ad ottenere la stessa maschera circolare usando
    # get_circular_mask e get_centered_circular_mask
    # la cmask che uso, e l'unica che funziona, ha
    # r_pix=120 e yc,xc=(240,308) che sono le coord del baricentro
    # ottenuto da quello degli attuatori al centro[63,64,75,76]
    # che sono diverse dalle cord del pixel centrale della parte di maschera a
    # False di imask==wfmask!

    def get_centred_circular_mask_wrt_interferometer_mask(self):
        # TODO: controllare che i dati a False siano una mappa rettangolare
        # prendo un generico pixel che sia a False per ricostruire base
        # e altezza della mappa rettangolare a False
        yFalsePixel = np.where(self._wfmask == False)[0][0]
        xFalsePixel = np.where(self._wfmask == False)[1][0]
        HeightInPixels = (self._wfmask[:, xFalsePixel] == False).sum()
        WidthInPixels = (self._wfmask[yFalsePixel, :] == False).sum()

        offsetX = (self._wfmask[yFalsePixel, 0:xFalsePixel] == True).sum()
        offsetY = (self._wfmask[0:yFalsePixel, xFalsePixel] == True).sum()
        # center of False map and origin of circular pupil in pixel
        yc0 = offsetY + 0.5 * HeightInPixels
        xc0 = offsetX + 0.5 * WidthInPixels
        MaxRadiusInPixel = min(WidthInPixels, HeightInPixels) * 0.5
        cmask = self.get_circular_mask(MaxRadiusInPixel, (yc0, xc0))

        return cmask

    def build_max_radius_and_pupil_in_imask(self):
        self._max_radius_in_imask, self._max_pupil_in_imask = self.get_centred_circular_mask_wrt_interferometer_mask()

    def _get_centered_circular_mask(self):
        '''
        centered wrt the intersection mask
        '''
        circular_mask = self._wfmask
        # center pixel coord of the full map
        central_ypix = self._wfmask.shape[0] // 2
        central_xpix = self._wfmask.shape[1] // 2
        # questo supponendo che il pixel centrale di tutto
        # il frame sia compreso dei dati a False...
        assert self._wfmask[central_ypix,
                            central_xpix] == False, f"Centred pixel of the full map is outside the useful(namely False) data!"
        HeightInPixels = (self._wfmask[:, central_xpix] == False).sum()
        WidthInPixels = (self._wfmask[central_ypix, :] == False).sum()

        offsetX = (self._wfmask[central_ypix,
                                0:self._wfmask.shape[1] // 2] == True).sum()
        offsetY = (self._wfmask[0:self._wfmask.shape[0] //
                                2, central_xpix] == True).sum()
        # center of False map and origin of circular mask in pixel
        yc0 = offsetY + HeightInPixels // 2
        xc0 = offsetX + WidthInPixels // 2
        RadiusInPixels = min(WidthInPixels, HeightInPixels) // 2
        for j in range(self._wfmask.shape[0]):
            for i in range(self._wfmask.shape[1]):
                Distanceinpixels = np.sqrt(
                    (j - yc0)**2 + (i - xc0)**2)
                if(Distanceinpixels <= RadiusInPixels):
                    circular_mask[j, i] = False
                else:
                    circular_mask[j, i] = True
        return circular_mask, RadiusInPixels, yc0, xc0


# da provare sul file cplm_all_fixed fatto il 17/3

class ModeGenerator():

    NORM_AT_THIS_CMD = 19  # such that wyko noise and saturation are avoided
    VISIBLE_AT_THIS_CMD = 19  # related cmd for actuators visibility in the given mask
    THRESHOLD_RMS = 0.5  # threshold for nasty actuators outside a given mask

    def __init__(self, cpla, mcl):
        self._cpla = cpla
        self._mcl = mcl
        self._n_of_act = self._cpla._wfs.shape[0]
        self._build_intersection_mask()

    def _build_intersection_mask(self):
        self._imask = reduce(lambda a, b: np.ma.mask_or(
            a, b), self._cpla._wfs[:, self.NORM_AT_THIS_CMD].mask)

    def _check_actuators_visibility(self, cmd=None):
        if cmd is None:
            cmd = self.VISIBLE_AT_THIS_CMD
        self._rms_wf = np.zeros(self._n_of_act)
        for act in range(self._n_of_act):
            self._rms_wf[act] = np.ma.array(data=self._cpla._wfs[act, cmd],
                                            mask=self._pupil_mask).std()

    def _show_actuators_visibility(self):
        plt.figure()
        plt.clf()
        plt.ion()
        plt.plot(self._rms_wf / 1.e-9, 'o', label='cmd=%d' %
                 self.VISIBLE_AT_THIS_CMD)
        plt.xlabel('#N actuator', size=25)
        plt.ylabel('Wavefront rms [nm]', size=25)
        plt.grid()
        plt.legend(loc='best')

    def _build_valid_actuators_list(self, cmd=None):

        self._check_actuators_visibility(cmd)
        self._acts_in_pupil = np.where(
            self._rms_wf > self.THRESHOLD_RMS * self._rms_wf.max())[0]

    def _normalize_influence_function(self, act):
        return (self._cpla._wfs[act, self.NORM_AT_THIS_CMD][self._pupil_mask == False] /
                self._mcl._deflection[act, self.NORM_AT_THIS_CMD]).data

    def _build_interaction_matrix(self):
        if self._acts_in_pupil is None:
            selected_act_list = self._cpla._actuators_list
        else:
            selected_act_list = self._acts_in_pupil
        self._im = np.column_stack([self._normalize_influence_function(
            act) for act in selected_act_list])

    def _build_reconstruction_matrix(self):
        self._rec = np.linalg.pinv(self._im)

    def compute_reconstructor(self, mask_obj=None):
        # TODO: check that mask.shape is equal to self._imask.shape
        # WARNING: zernike_generator uses the pupil mask as the object!!!
        # while self._pupil_mask is a bool array!
        if mask_obj is None:
            mask = self._imask  # bool array
        else:
            self._pupil_mask_obj = mask_obj
            mask = self._pupil_mask_obj.mask()
        assert self._imask.shape == mask.shape, f"mask has not the same dimension of self._imask!\nGot:{mask.shape}\nShould be:{self._imask.shape}"
        self._pupil_mask = np.ma.mask_or(self._imask, mask)

        self._build_valid_actuators_list()
        self._build_interaction_matrix()
        self._build_reconstruction_matrix()

    def generate_mode(self, wfmap):
        self._wfmode = np.ma.array(data=wfmap, mask=self._pupil_mask)

    def generate_zernike_mode(self, j, AmpInMeters):
        zg = ZernikeGenerator(self._pupil_mask_obj)
        self._wfmode = np.zeros(self._pupil_mask.shape)
        self._wfmode = np.ma.array(data=self._wfmode, mask=self._pupil_mask)
        z_mode = zg.getZernike(j)
        a = (z_mode.mask == False).sum()
        b = (self._wfmode.mask == False).sum()
        assert a == b, f"zerike valid points: {a}  wfmode valid points: {b}\nShould be equal!"
        # should be useless
        unmasked_index_wf = np.ma.where(self._wfmode.mask == False)
        unmasked_index_zernike = np.ma.where(z_mode.mask == False)
        self._wfmode[unmasked_index_wf[0], unmasked_index_wf[1]
                     ] = z_mode.data[unmasked_index_zernike[0], unmasked_index_zernike[1]]
        self._wfmode = self._wfmode * AmpInMeters

    def generate_tilt(self):
        self._wfmode = np.tile(np.linspace(-100e-9, 100e-9, 640), (486, 1))
        self._wfmode = np.ma.array(data=self._wfmode, mask=self._pupil_mask)

    def get_position_cmds_from_wf(self, wfmap=None):
        if wfmap is None:
            wfmap = self._wfmode
        pos = np.dot(self._rec, wfmap[self._pupil_mask == False])
        # check and clip cmds
        # should I clip voltage or stroke cmds?
        # act's stroke increases when moved with its neighbour
        for idx in range(len(pos)):
            max_stroke = np.max(
                self._mcl._deflection[self._acts_in_pupil[idx]])
            min_stroke = np.min(
                self._mcl._deflection[self._acts_in_pupil[idx]])
            if(pos[idx] > max_stroke):
                pos[idx] = max_stroke
                print('act%d reached max stroke' % self._acts_in_pupil[idx])
            if(pos[idx] < min_stroke):
                pos[idx] = min_stroke
                print('act%d reached min stroke' % self._acts_in_pupil[idx])
        return pos

    def build_fitted_wavefront(self, wfmap=None):
        if wfmap is None:
            wfmap = self._wfmode
        pos_from_wf = self.get_position_cmds_from_wf(wfmap)
        self._wffitted = np.zeros(
            (self._cpla._wfs.shape[2], self._cpla._wfs.shape[3]))
        self._wffitted[self._pupil_mask == False] = np.dot(
            self._im, pos_from_wf)
        self._wffitted = np.ma.array(
            data=self._wffitted, mask=self._pupil_mask)

    def plot_generated_and_expected_wf(self):
        plt.figure()
        plt.clf()
        plt.imshow(self._wfmode / 1.e-9)
        plt.colorbar(label='[nm]')
        plt.title('Generated Mode', size=25)

        plt.figure()
        plt.clf()
        plt.imshow(self._wffitted / 1.e-9)
        plt.colorbar(label='[nm]')
        plt.title('Fitted Mode', size=25)
        plt.figure()
        plt.clf()
        plt.imshow((self._wffitted - self._wfmode) / 1.e-9)
        plt.colorbar(label='[nm]')
        plt.title('Mode difference', size=25)

        print("Expectations:")
        amp = self._wffitted.std()
        amp = amp / 1.e-9
        print("mode amplitude: %g nm rms " % amp)
        fitting_error = (self._wffitted - self._wfmode).std()
        fitting_error = fitting_error / 1.e-9
        print("fitting error: %g nm rms " % fitting_error)

    def vector_to_map(self, wf_vector):
        mappa = np.zeros(
            (self._cpla._wfs.shape[2], self._cpla._wfs.shape[3]))
        mappa[self._pupil_mask == False] = wf_vector
        return np.ma.array(data=mappa, mask=self._pupil_mask)


class ModeMeasurer():
    # fraction of valid pixels in wf measures: avoids nasty maps
    THRESHOLD_RATIO = 0.99
    fnpath = 'prova/static_zernike_modes/'
    ffmt = '.png'
    AmpInNanometer = 100
    TIME_OUT = 10  # s

    def __init__(self, interferometer, mems_deformable_mirror):
        self._interf = interferometer
        self._bmc = mems_deformable_mirror

    def execute_measure(self, mcl, mg, pos=None):
        if pos is None:
            pos = mg.get_position_cmds_from_wf()

        flat_cmd = np.zeros(self._bmc.get_number_of_actuators())
        self._bmc.set_shape(flat_cmd)
        ref_cmd = self._bmc.get_reference_shape()

        expected_valid_points = (mg._pupil_mask == False).sum()

        wfflat = self._interf.wavefront(timeout_in_sec=self.TIME_OUT)
        wfflat = np.ma.array(data=wfflat, mask=mg._pupil_mask)
        # avoid nasty wf maps
        measured_valid_points = (wfflat.mask == False).sum()
        ratio = measured_valid_points / expected_valid_points
        while(ratio < self.THRESHOLD_RATIO):
            print('Warning: Nasty map acquired!Reloading...')
            wfflat = self._interf.wavefront(timeout_in_sec=self.TIME_OUT)
            wfflat = np.ma.array(data=wfflat, mask=mg._pupil_mask)
            measured_valid_points = (wfflat.mask == False).sum()
            ratio = measured_valid_points / expected_valid_points

        act_list = mg._acts_in_pupil
        cmd = np.zeros(self._bmc.get_number_of_actuators())
        # TODO: clip voltage!
        for idx in range(len(pos)):
            cmd[act_list[idx]] = mcl.p2c(act_list[idx], pos[idx])
            # giustamente se clippo in tensione...
            # ValueError: A value in x_new is above the interpolation range.
            # volt_control = cmd[act_list[idx]] + ref_cmd[act_list[idx]]
            # if(volt_control > 1.):
            #     print('act%d reaches min stroke!' % act_list[idx])
            #     cmd[act_list[idx]] = 1. - ref_cmd[act_list[idx]]
            # if(volt_control < 0.):
            #     print('act%d reaches max stroke!' % act_list[idx])
            #     cmd[act_list[idx]] = 0. - ref_cmd[act_list[idx]]
        self._bmc.set_shape(cmd)
        #_get_wavefront_flat_subtracted
        wfflatsub = self._interf.wavefront(
            timeout_in_sec=self.TIME_OUT) - wfflat
        self._wfmeas = wfflatsub - np.ma.median(wfflatsub)
        self._wfmeas = np.ma.array(data=self._wfmeas, mask=mg._pupil_mask)
        # avoid nasty wf maps
        measured_valid_points = (self._wfmeas.mask == False).sum()
        ratio = measured_valid_points / expected_valid_points
        while(ratio < self.THRESHOLD_RATIO):
            print('Warning: Nasty map acquired!Reloading...')
            wfflatsub = self._interf.wavefront(
                timeout_in_sec=self.TIME_OUT) - wfflat
            self._wfmeas = wfflatsub - np.ma.median(wfflatsub)
            self._wfmeas = np.ma.array(data=self._wfmeas, mask=mg._pupil_mask)
            measured_valid_points = (self._wfmeas.mask == False).sum()
            ratio = measured_valid_points / expected_valid_points

    def plot_expected_and_measured_mode(self, wfexpected):
        plt.figure()
        plt.clf()
        plt.ion()
        plt.imshow(self._wfmeas / 1.e-9)
        plt.colorbar(label='[nm]')
        plt.title('Observed Mode', size=25)
        plt.figure()
        plt.clf()
        plt.ion()
        plt.imshow((self._wfmeas - wfexpected) / 1.e-9)
        plt.colorbar(label='[nm]')
        plt.title('Difference Observed-Expected', size=25)
        print("Observation:")
        amp_mode = self._wfmeas.std()
        amp_mode = amp_mode / 1.e-9
        print("mode amplitude: %g nm rms " % amp_mode)
        fitting_meas_error = (self._wfmeas - wfexpected).std()
        fitting_meas_error = fitting_meas_error / 1.e-9
        print("fitting error: %g nm rms " % fitting_meas_error)

    def _test_modes_measure(self, mcl, mg, Nmodes):
        j_modes = np.arange(2, Nmodes + 1, dtype='int')
        Nmodes = len(j_modes)
        a_j = self.AmpInNanometer * 1.e-9
        A = self.AmpInNanometer
        expected_modes_stat = np.zeros((Nmodes, 1, 2))
        measured_modes_stat = np.zeros((Nmodes, 1, 2))

        for idx, j in enumerate(j_modes):
            j = int(j)
            mg.generate_zernike_mode(j, a_j)
            mg.build_fitted_wavefront()
            exp_wf_rms = mg._wfmode.std()
            exp_fitting_error = (mg._wffitted - mg._wfmode).std()

            self.execute_measure(mcl, mg)
            meas_wf_rms = self._wfmeas.std()
            meas_fitting_error = (self._wfmeas - mg._wfmode).std()

            plt.ioff()
            plt.figure()
            plt.clf()
            plt.imshow(mg._wfmode / 1.e-9)
            plt.colorbar(label='[nm]')
            plt.title('Generated Mode')
            plt.savefig(fname=self.fnpath + 'Z%d' %
                        j + '_1gen' + '_A%d' % A + self.ffmt, bbox_inches='tight')
            plt.close()
            plt.figure()
            plt.clf()
            plt.imshow((mg._wffitted) / 1.e-9)
            plt.colorbar(label='[nm]')
            plt.title('Fitted')
            plt.savefig(fname=self.fnpath + 'Z%d' %
                        j + '_2fitted' + '_A%d' % A + self.ffmt, bbox_inches='tight')
            plt.close()
            plt.figure()
            plt.clf()
            plt.imshow((mg._wffitted - mg._wfmode) / 1.e-9)
            plt.colorbar(label='[nm]')
            a = exp_fitting_error / 1.e-9
            plt.title('Fitted - Generated: rms %g nm' % a)
            plt.savefig(fname=self.fnpath + 'Z%d' %
                        j + '_3fitgendiff' + '_A%d' % A + self.ffmt, bbox_inches='tight')
            plt.close()
            plt.figure()
            plt.clf()
            plt.imshow((self._wfmeas) / 1.e-9)
            plt.colorbar(label='[nm]')
            plt.title('Observed Mode')
            plt.savefig(fname=self.fnpath + 'Z%d' %
                        j + '_4obs' + '_A%d' % A + self.ffmt, bbox_inches='tight')
            plt.close()
            plt.figure()
            plt.clf()
            plt.imshow((self._wfmeas - mg._wfmode) / 1.e-9)
            plt.colorbar(label='[nm]')
            a = meas_fitting_error / 1.e-9
            plt.title('Observed - Generated: rms %g nm' % a)
            plt.savefig(fname=self.fnpath + 'Z%d' %
                        j + '_5obsgendiff' + '_A%d' % A + self.ffmt, bbox_inches='tight')
            plt.close()

            expected_modes_stat[idx, 0, 0] = exp_wf_rms
            expected_modes_stat[idx, 0, 1] = exp_fitting_error
            measured_modes_stat[idx, 0, 0] = meas_wf_rms
            measured_modes_stat[idx, 0, 1] = meas_fitting_error

        return expected_modes_stat, measured_modes_stat


class SuitableActuatorsInPupilAnalyzer():
    # TODO: data la pupilla e dati i modi che si vogliono generare,
    # determinare la lista degli attuatori/threshold che minimizzano il fitting error osservato
    # dei modi che si vogliono generare
    # to be continued...

    THRESHOLD_SPAN = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5])
    Aj_SPAN = 100.e-9 * np.arange(1, 6)  # meters

    def __init__(self, mcl, mg, mm, pupil_mask_obj):
        self._calibration = mcl
        self._mode_generator = mg
        self._mode_measurer = mm
        self._pupil_mask = pupil_mask_obj

    def _test_measure(self, NumOfZmodes):
        # except Z1
        jmodes = np.arange(2, NumOfZmodes + 1)
        self._generated_jmodes = jmodes
        num_of_gen_modes = len(jmodes)
        num_of_threshold = len(self.THRESHOLD_SPAN)
        num_of_ampj = len(self.Aj_SPAN)
        frame_shape = self._pupil_mask.mask().shape
        # per un dato threshold, modo e ampiezza misuro il
        # fitting error aspettato e misurato
        self._fitting_sigmas = np.zeros(
            (num_of_threshold, num_of_ampj, num_of_gen_modes, 2))
        self._wfs_gen = np.ma.zeros(
            (num_of_threshold, num_of_ampj, num_of_gen_modes, frame_shape[0], frame_shape[1]))
        self._wfs_fitted = np.ma.zeros(
            (num_of_threshold, num_of_ampj, num_of_gen_modes, frame_shape[0], frame_shape[1]))
        self._wfs_meas = np.ma.zeros(
            (num_of_threshold, num_of_ampj, num_of_gen_modes, frame_shape[0], frame_shape[1]))

        self._valid_act_per_thres = []

        for thres_idx, threshold in enumerate(self.THRESHOLD_SPAN):
            print("Threshold set to:%g" % threshold)
            self._mode_generator.THRESHOLD_RMS = threshold
            self._mode_generator.compute_reconstructor(
                mask_obj=self._pupil_mask)
            self._valid_act_per_thres.append(
                self._mode_generator._acts_in_pupil)

            for amp_idx, aj in enumerate(self.Aj_SPAN):
                print("Generating Zmodes with amplitude[m] set to: %g" % aj)
                for j_idx, j in enumerate(jmodes):

                    self._mode_generator.generate_zernike_mode(int(j), aj)
                    self._mode_generator.build_fitted_wavefront()
                    # expected_amplitude = (self._mode_generator._wffitted).std()
                    expected_fitting_error = (
                        self._mode_generator._wffitted - self._mode_generator._wfmode).std()
                    self._mode_measurer.execute_measure(
                        self._calibration, self._mode_generator)
                    # measured_amplitude = (self._mode_measurer._wfmeas).std()
                    measured_fitting_error = (
                        self._mode_measurer._wfmeas - self._mode_generator._wfmode).std()
                    self._fitting_sigmas[thres_idx, amp_idx,
                                         j_idx] = expected_fitting_error, measured_fitting_error
                    self._wfs_gen[thres_idx, amp_idx,
                                  j_idx] = self._mode_generator._wfmode
                    self._wfs_fitted[thres_idx, amp_idx,
                                     j_idx] = self._mode_generator._wffitted
                    self._wfs_meas[thres_idx, amp_idx,
                                   j_idx] = self._mode_measurer._wfmeas

    def _show_fitting_errors_for(self, threshold, amplitude, jmode):
        thres_idx = np.where(self.THRESHOLD_SPAN == threshold)[0][0]
        amp_idx = np.where(self.Aj_SPAN == amplitude)[0][0]
        j_idx = np.where(self._generated_jmodes == jmode)[0][0]
        print("Threshold = {}; Amplitude[m] = {}; Mode = Z{}".format(
            threshold, amplitude, jmode))
        print("Expected fitting error[m] = {} \nMeasured fitting error[m] = {} ".format(
            self._fitting_sigmas[thres_idx, amp_idx, j_idx, 0], self._fitting_sigmas[thres_idx, amp_idx, j_idx, 1]))

    def _test_recontruct_zmodes_up_to(self, NumOfZmodes, threshold, AmplitudeInMeters):
        '''
        per fissato threshold e AmplitudeInMeters, ricostruisce 
        i primi NumOfZmodes modi di zernike che il mems, potenzialmente,
        e in grado di riprodurre
        '''
        jmodes = np.arange(2, NumOfZmodes + 1)
        self._expected_fitting_error = np.zeros(len(jmodes))
        self._mode_generator.THRESHOLD_RMS = threshold
        self._mode_generator.compute_reconstructor(
            mask_obj=self._pupil_mask)
        for idx, j in enumerate(jmodes):
            self._mode_generator.generate_zernike_mode(
                int(j), AmplitudeInMeters)
            self._mode_generator.build_fitted_wavefront()
            self._expected_fitting_error[idx] = (
                self._mode_generator._wffitted - self._mode_generator._wfmode).std()
        print(self._mode_generator._acts_in_pupil)
        print('Suitable actuators #N = %d' %
              len(self._mode_generator._acts_in_pupil))
        plt.figure()
        plt.clf()
        plt.plot(jmodes, self._expected_fitting_error /
                 1.e-9, 'bo-', label='expected')
        plt.title('expected fitting error for: amp = %g[m]' %
                  AmplitudeInMeters + ' threshold = %g' % threshold, size=25)
        plt.xlabel(r'$Z_j$', size=25)
        plt.ylabel(r'$WF_{fit}-WF_{gen} rms [nm]$', size=25)
        plt.grid()
        plt.legend(loc='best')
        return self._expected_fitting_error

    def _test_compute_exp_fitting_err_up_to(self, NumOfZmodes, AmplitudeInMeters):
        '''
        voglio capire fino a quale modo Zj potenzialmente il mems e in grado di riprodurre
        al variare del numero di attuatori(threshold della visibilita) e al variare dell
        ampiezza del modo
        '''
        jmodes = np.arange(2, NumOfZmodes + 1)
        num_of_jmodes = len(jmodes)
        num_of_threshold = len(self.THRESHOLD_SPAN)
        num_of_valid_acts = np.zeros(len(self.THRESHOLD_SPAN))
        fitting_error = np.zeros((num_of_threshold, num_of_jmodes))
        for thres_idx, threshold in enumerate(self.THRESHOLD_SPAN):
            self._mode_generator.THRESHOLD_RMS = threshold
            self._mode_generator.compute_reconstructor(
                mask_obj=self._pupil_mask)
            num_of_valid_acts[thres_idx] = len(
                self._mode_generator._acts_in_pupil)
            for j_idx, j in enumerate(jmodes):
                self._mode_generator.generate_zernike_mode(
                    int(j), AmplitudeInMeters)
                self._mode_generator.build_fitted_wavefront()
                fitting_error[thres_idx, j_idx] = (
                    self._mode_generator._wffitted - self._mode_generator._wfmode).std()

        plt.figure()
        plt.clf()
        plt.title('expected fitting error for: amp = %g[m]' %
                  AmplitudeInMeters)
        self._test_plot_exp_fitting_err(fitting_error, num_of_valid_acts)

        return fitting_error, num_of_valid_acts

    def _test_plot_exp_fitting_err(self, fitting_error, num_of_valid_acts):

        jmodes = 2 + np.arange(fitting_error.shape[1])
        for thres_idx, thres in enumerate(self.THRESHOLD_SPAN):
            plt.plot(jmodes, fitting_error[thres_idx] /
                     1.e-9, 'o-', label='thres=%g' % thres + '#Nact=%d' % num_of_valid_acts[thres_idx])
        plt.xlabel(r'$Z_j$', size=25)
        plt.ylabel(r'$WF_{fit}-WF_{gen} rms [nm]$', size=25)
        plt.grid()
        plt.legend(loc='best')

    def _plot_fitting_errors_for(self, threshold, amplitude):
        thres_idx = np.where(self.THRESHOLD_SPAN == threshold)[0][0]
        amp_idx = np.where(self.Aj_SPAN == amplitude)[0][0]
        plt.figure()
        plt.clf()

        exp_fitting_err = self._fitting_sigmas[thres_idx, amp_idx, :, 0]
        meas_fitting_err = self._fitting_sigmas[thres_idx, amp_idx, :, 1]
        exp_fitting_err = exp_fitting_err / 1.e-9
        meas_fitting_err = meas_fitting_err / 1.e-9

        plt.plot(self._generated_jmodes, exp_fitting_err,
                 'bo-', label='expected')
        plt.plot(self._generated_jmodes, meas_fitting_err,
                 'ro-', label='measured')
        plt.legend(loc='best')
        plt.title('fitting error: amp[m]=%g' %
                  amplitude + ' threshold=%g' % threshold, size=25)
        plt.xlabel('Zmode j index', size=25)
        plt.ylabel('WF rms [nm]', size=25)

    def save_results(self, fname):
        # syntax error see astropy
        hdr = fits.Header()
        #hdr['CMASK'] = self._pupil_mask
        #hdr['AMP_INM'] = self.Aj_SPAN
        fits.writeto(fname, self._fitting_sigmas, hdr)
        fits.append(fname, self.THRESHOLD_SPAN)
        fits.append(fname, self.Aj_SPAN)
        fits.append(fname, self._generated_jmodes)
        fits.append(fname, self._pupil_mask.mask().astype(int))
        fits.append(fname, self._wfs_gen)
        fits.append(fname, self._wfs_fitted)
        fits.append(fname, self._wfs_meas)
        #fits.append(fname, self._valid_act_per_thres)

    @staticmethod
    def load(fname):
        header = fits.getheader(fname)
        hduList = fits.open(fname)
        sigma_data = hduList[0].data
        thres_data = hduList[1].data
        amp_data = hduList[2].data
        jmodes_data = hduList[3].data

        cmask2d = hduList[4].data.astype(bool)
        wfs_gen_data = hduList[5].data
        wfs_fit_data = hduList[6].data
        wfs_meas_data = hduList[7].data

        wfs_mask = np.ma.zeros(wfs_gen_data.shape)
        wfs_mask[:, :, :] = cmask2d
        ma_wfsgen = np.ma.array(data=wfs_gen_data, mask=wfs_mask)
        ma_wfsfit = np.ma.array(data=wfs_fit_data, mask=wfs_mask)
        ma_wfsmeas = np.ma.array(data=wfs_meas_data, mask=wfs_mask)

        #valid_act_data = hduList[4].data
        return{'sigmas': sigma_data,
               'thres': thres_data,
               'amp': amp_data,
               'jmode': jmodes_data,
               'wfsmask': wfs_mask,
               'wfsgen': ma_wfsgen,
               'wfsfit': ma_wfsfit,
               'wfsmeas': ma_wfsmeas
               #'valid_acts': valid_act_data
               }


class _test_saipa_load():
    def __init__(self, fname):
        res = SuitableActuatorsInPupilAnalyzer.load(fname)
        self._sigmas = res['sigmas']
        self._threshold_span = res['thres']
        self._amplitude_span = res['amp']
        self._jmodes = res['jmode']
        self._wfs_mask = res['wfsmask']
        self._wfs_gen = res['wfsgen']
        self._wfs_fitted = res['wfsfit']
        self._wfs_meas = res['wfsmeas']
        self._wfs_mask = res['wfsmask']
        #self._act_list_per_thres = res['valid_acts']

    def _test_plot_meas_vs_exp_fitting_errors(self):

        for amp_idx, amplitude in enumerate(self._amplitude_span):
            plt.figure()
            plt.clf()
            for thres_idx, threshold in enumerate(self._threshold_span):

                exp_fitting_err = self._sigmas[thres_idx, amp_idx, :, 0]
                meas_fitting_err = self._sigmas[thres_idx,
                                                amp_idx, :, 1]
                exp_fitting_err = exp_fitting_err / 1.e-9
                meas_fitting_err = meas_fitting_err / 1.e-9

                plt.plot(self._jmodes, exp_fitting_err,
                         '.--', label='exp: threshold=%g' % threshold)
                plt.plot(self._jmodes, meas_fitting_err,
                         'o-', label='meas: threshold=%g' % threshold, color=plt.gca().lines[-1].get_color())

            plt.legend(loc='best')
            plt.title('fitting error: amp[m]=%g' %
                      amplitude, size=25)
            plt.xlabel(r'$Z_j$', size=25)
            plt.ylabel(r'$WF_-WF_{gen} rms [nm]$', size=25)
            plt.grid()


def provarec(cpla, mcl):
    # maschera intersezione di tutte le maschere delle wf map misurate
    imask = reduce(lambda a, b: np.ma.mask_or(a, b), cpla._wfs[:, 13].mask)

    # normalizzo la mappa di fase dell'attuatore act a max==1
    # scelgo il comando 13: è una deformata di circa 500nm (quindi ben sopra al
    # rumore dell'interferometro)
    # e non troppo grossa da saturare l'interferometro: per tutti i 140
    # attuatori la mappa del comando 13 non presenta "buchi"
    #
    # la funzione ritorna un vettore contenente i valori del wf nei punti
    # dentro la maschera imask
    def normalizeif(act):
        return (cpla._wfs[act, 13][imask == False] / mcl._deflection[act, 13]).data

    # creo una "matrice di interazione" giustapponendo in colonna i vettori
    # normalizzati
    im = np.column_stack([normalizeif(i) for i in cpla._actuators_list])

    # pseudo inversa della matrice di interazione
    rec = np.linalg.pinv(im)

    # questo prodotto matriciale fra rec e una mappa di fase qualsiasi restituisce
    # le 140 posizioni in cui comandare gli attuatori per ottenere wfmap
    def wf2pos(wfmap):
        return np.dot(rec, wfmap[imask == False])

    # creo un tilt (da -100 a 100nm lungo ogni riga, tutte le righe sono uguali
    wftilt = np.tile(np.linspace(-100e-9, 100e-9, 640), (486, 1))
    # lo converto in masked_array per coerenza col resto
    wftilt = np.ma.array(data=wftilt, mask=imask)

    # postilt è il comando da dare agli attuatori per ottenere wftilt
    postilt = wf2pos(wftilt)
    # bisognerebbe controllare che nessun elemento sia troppo grande,
    # altrimenti lo specchio non riesce a fare la deformata richiesta

    # wffitted è la mappa di fase che mi aspetto di ottenere davvero:
    # è il miglior fit che lo specchio riesce a fare di wftilt
    wffitted = np.zeros((cpla._wfs.shape[2], cpla._wfs.shape[3]))
    wffitted[imask == False] = np.dot(im, postilt)
    wffitted = np.ma.array(data=wffitted, mask=imask)

    print("mode amplitude: %g nm rms " % wftilt.std())
    fitting_error = (wffitted - wftilt).std()
    print("fitting error: %g nm rms " % fitting_error)

    # posso rifarlo con un modo ad alta frequenza (tipo un seno che oscilla 15
    # volte in 640 px)
    wfsin = np.tile(100e-9 * np.sin(2 * np.pi / 43 * np.arange(640)), (486, 1))
    wfsin = np.ma.array(data=wfsin, mask=imask)

    possin = wf2pos(wfsin)

    sinfitted = np.zeros((486, 640))
    sinfitted[imask == False] = np.dot(im, possin)
    sinfitted = np.ma.array(data=sinfitted, mask=imask)

    # il fitting error su questo modo è 2nm rms
    fitting_error_sin = (sinfitted - wfsin).std()

    print("mode amplitude sin: %g nm rms " % wfsin.std())
    print("fitting error: %g nm rms " % fitting_error_sin)


class TestSvd():

    def __init__(self, mg):
        self.mg = mg
        # mg = ModeGenerator
        self.u, self.s, self.vh = np.linalg.svd(
            self.mg._im, full_matrices=False)

    def autovettori(self, eigenvalue_index):
        wf = np.zeros((486, 640))
        wf[self.mg._imask == False] = np.dot(
            self.mg._im, self.vh.T[:, eigenvalue_index])
        return np.ma.array(wf, mask=self.mg._imask)

    def rec(self, eigenvalue_to_use):
        large = np.zeros(self.s.shape).astype(bool)
        large[eigenvalue_to_use] = True
        s = np.divide(1, self.s, where=large)
        s[~large] = 0
        res = np.matmul(np.transpose(self.vh), np.multiply(
            s[..., np.newaxis], np.transpose(self.u)))
        return res

    def animate(self, interval=100):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        fig = plt.figure()

        self._ani_index = 0
        im = plt.imshow(self.autovettori(0), animated=True)

        def updatefig(*args):
            self._ani_index += 1
            im.set_array(self.autovettori(self._ani_index % 140))
            plt.title("Eigenmode %d" % self._ani_index)
            return im,

        self._ani = animation.FuncAnimation(
            fig, updatefig, interval=interval, blit=True)
        plt.show()


class TestRepeatedMeasures():
    ffmt = '.fits'
    fpath = 'prova/misure_ripetute/trm_'

    def __init__(self, interferometer, mems_deformable_mirror):
        self._interf = interferometer
        self._bmc = mems_deformable_mirror

    def _execute_repeated_measure(self, act_list, Ntimes):

        act_list = np.array(act_list)
        n_acts = len(act_list)

        cplm = CommandToPositionLinearizationMeasurer(self._interf, self._bmc)

        cplm.NUMBER_STEPS_VOLTAGE_SCAN = 20
        for act_idx, act in enumerate(act_list):
            print('Repeated measures for act%d' % int(act))
            for times in np.arange(Ntimes):
                print('%dtimes:' % times)
                cplm.execute_command_scan([int(act)])
                fname = self.fpath + \
                    'act%d' % int(act) + 'time%d' % times + self.ffmt
                cplm.save_results(fname)

    def _analyze_measure(self, act_list, n_steps_volt_scan, Ntimes):
        '''
        legge i file cplm relativi alle misure ripetute su ogni attuatote,
        calcola la media delle deflessioni ottenute per ciascun comando
        in tensione applicato e le salva sul file fits che dovra essere
        caricato da un oggetto mcl
        '''
        if act_list is None:
            act_list = np.arange(self._bmc.get_number_of_actuators())
        else:
            act_list = np.array([act_list])

        n_acts = len(act_list)
        self._Ncpla_cmd_vector = np.zeros((Ntimes, n_acts, n_steps_volt_scan))
        self._Ncpla_deflection = np.zeros((Ntimes, n_acts, n_steps_volt_scan))
        for times in np.arange(Ntimes):
            print('%dtimes:' % times)
            for act_idx, act in enumerate(act_list):
                print('Loading act#%d from file' % int(act))
                fname = self.fpath + \
                    'act%d' % int(act) + 'time%d' % times + self.ffmt
                cpla = CommandToPositionLinearizationAnalyzer(fname)
                cpla._compute_maximum_deflection()
                # print(cpla._cmd_vector.shape)
                # loaded cpla._cmd_vector has(1,20) shape
                self._Ncpla_cmd_vector[times,
                                       act_idx] = cpla._cmd_vector[0]
                self._Ncpla_deflection[times,
                                       act_idx] = cpla._max_deflection[0]
        print('Mean deflections estimation...')
        self._deflection_mean = np.zeros((n_acts, n_steps_volt_scan))
        self._deflection_err = np.zeros((n_acts, n_steps_volt_scan))
        for act_idx, act in enumerate(act_list):
            print('Loading act#%d' % int(act))
            for cmd_idx in np.arange(n_steps_volt_scan):
                self._deflection_mean[act_idx, cmd_idx] = np.mean(
                    self._Ncpla_deflection[:, act_idx, cmd_idx])
                self._deflection_err[act_idx, cmd_idx] = np.std(
                    self._Ncpla_deflection[:, act_idx, cmd_idx])
        print('Creating mcl object...')
        self._mcl = MemsCommandLinearization(
            act_list, self._Ncpla_cmd_vector[0], self._deflection_mean, self._bmc.get_reference_shape_tag())
        # in realta e inutile perche poi devo fare il load e ricalcolare l
        # interpolazione
        self._mcl._create_interpolation_test()
        self._mcl.save(self.fpath + 'mcl_all' + self.ffmt)
        print('mcl object saved!')

    def _collapse_all_measured_wfs(self, act_list, n_steps_volt_scan, Ntimes):
        '''
        vorrei legge i file cplm relativi alle misure ripetute su ogni attuatote,
        fare la media delle mappe ottunte per il dato act e cmd, in modo da salvare
        su file un solo wfmap per ogni act e comando       
        '''
        if act_list is None:
            act_list = np.arange(self._bmc.get_number_of_actuators())
        else:
            act_list = np.array([act_list])
        n_acts = len(act_list)
        # TODO: prendere il frame direttamente dall interferometro
        frame_shape = np.array([486, 640])
        wfs_temp = np.ma.zeros(
            (Ntimes, n_steps_volt_scan, frame_shape[0], frame_shape[1]))
        self._collapsed_wfs = np.ma.zeros(
            (n_acts, n_steps_volt_scan, frame_shape[0], frame_shape[1]))
        # self._collapsed_sigma_wfs = np.ma.zeros(
        #     (n_acts, n_steps_volt_scan, frame_shape[0], frame_shape[1]))

        for act_idx, act in enumerate(act_list):
            for times in np.arange(Ntimes):
                fname = self.fpath + \
                    'act%d' % int(act) + 'time%d' % times + self.ffmt
                cpla = CommandToPositionLinearizationAnalyzer(fname)
                for cmd_idx in np.arange(n_steps_volt_scan):
                    wfs_temp[times, cmd_idx] = cpla._wfs[act_idx, cmd_idx]
            print('time to collapse!')
            for cmd_idx in np.arange(n_steps_volt_scan):
                print('cmd_idx=%d' % cmd_idx)
                for yi in np.arange(frame_shape[0]):
                    for xi in np.arange(frame_shape[1]):
                        self._collapsed_wfs[act_idx, cmd_idx, yi,
                                            xi] = wfs_temp[:, cmd_idx, yi, xi].mean()
                        # self._collapsed_sigma_wfs[act_idx, cmd_idx, yi,
                        # xi] = wfs_temp[:, cmd_idx, yi, xi].std()
        print('collapsed!')

        cplm = CommandToPositionLinearizationMeasurer(self._interf, self._bmc)
        cplm.NUMBER_STEPS_VOLTAGE_SCAN = n_steps_volt_scan
        cplm._wfs = np.ma.zeros(
            (n_acts, n_steps_volt_scan, frame_shape[0], frame_shape[1]))
        cplm._wfs = self._collapsed_wfs
        cplm._cmd_vector = cpla._cmd_vector
        cplm._actuators_list = cpla._actuators_list
        cplm._reference_cmds = self._bmc.get_reference_shape()
        cplm._reference_tag = self._bmc.get_reference_shape_tag()
        cplm.NUMBER_WAVEFRONTS_TO_AVERAGE = 1
        fname = self.fpath + 'cplm_all_mean' + self.ffmt
        cplm.save_results(fname)

    def _plot_deflections_vs_cmd(self, act, Ntimes):

        plt.figure()
        plt.clf()
        for times in np.arange(Ntimes):
            plt.plot(self._Ncpla_cmd_vector[times, act],
                     self._Ncpla_deflection[times, act] / 1.e-9, 'o', label='%dtime' % times)
        plt.grid()
        plt.legend(loc='best')
        plt.title('act=%d' % int(act))


class InfluenceFunctionMeasurer():
    ffmt = '.fits'
    fpath = 'prova/misure_ifs/ifm_'
    NUMBER_WAVEFRONTS_TO_AVERAGE = 1
    NUMBER_STEPS_VOLTAGE_SCAN = 1
    TIME_OUT = 10  # sec

    def __init__(self, interferometer, mems_deformable_mirror):
        self._interf = interferometer
        self._bmc = mems_deformable_mirror
        self._n_acts = self._bmc.get_number_of_actuators()
        self._wfflat = None

    def _get_zero_command_wavefront(self):
        if self._wfflat is None:
            cmd = np.zeros(self._n_acts)
            self._bmc.set_shape(cmd)
            self._wfflat = self._interf.wavefront(
                self.NUMBER_WAVEFRONTS_TO_AVERAGE, timeout_in_sec=self.TIME_OUT)
        return self._wfflat

    def execute_ifs_measure(self, mcl, pos):

        act_list = np.arange(self._n_acts)

        self._actuators_list = np.array(act_list)
        n_acts_to_meas = len(self._actuators_list)

        wfflat = self._get_zero_command_wavefront()

        self._reference_cmds = self._bmc.get_reference_shape()
        self._reference_tag = self._bmc.get_reference_shape_tag()

        self._cmd_vector = np.zeros((n_acts_to_meas,
                                     self.NUMBER_STEPS_VOLTAGE_SCAN))

        self._wfs = np.ma.zeros(
            (n_acts_to_meas, self.NUMBER_STEPS_VOLTAGE_SCAN,
             wfflat.shape[0], wfflat.shape[1]))

        N_pixels = self._wfs.shape[2] * self._wfs.shape[3]
        for act_idx, act in enumerate(self._actuators_list):
            self._cmd_vector[act_idx] = mcl.linear_p2c(int(act), pos)
            for cmd_idx, cmdi in enumerate(self._cmd_vector[act_idx]):
                print("Act:%d - command %g" % (act, cmdi))
                cmd = np.zeros(self._n_acts)
                cmd[act] = cmdi
                self._bmc.set_shape(cmd)
                self._wfs[act_idx, cmd_idx, :,
                          :] = self._get_wavefront_flat_subtracted()
                masked_pixels = self._wfs[act_idx, cmd_idx].mask.sum()
                masked_ratio = masked_pixels / N_pixels
                if masked_ratio > 0.7829:
                    print('Warning: Bad measure acquired for: act%d' %
                          act_idx + ' cmd_idx %d' % cmd_idx)
                    self._avoid_saturated_measures(
                        masked_ratio, act_idx, cmd_idx, N_pixels)

    def _avoid_saturated_measures(self, masked_ratio, act_idx, cmd_idx, N_pixels):

        while masked_ratio > 0.7829:
            self._wfs[act_idx, cmd_idx, :,
                      :] = self._get_wavefront_flat_subtracted()
            masked_pixels = self._wfs[act_idx, cmd_idx].mask.sum()
            masked_ratio = masked_pixels / N_pixels

        print('Repeated measure completed!')

    def _get_wavefront_flat_subtracted(self):
        dd = self._interf.wavefront(
            self.NUMBER_WAVEFRONTS_TO_AVERAGE, timeout_in_sec=self.TIME_OUT) - self._get_zero_command_wavefront()
        return dd - np.ma.median(dd)

    def _reset_flat_wavefront(self):
        self._wfflat = None

    def check_mask_coverage(self, ratio=False):
        masked_pixels = np.array([self._wfs[a, i].mask.sum() for a in range(
            self._wfs.shape[0]) for i in range(self._wfs.shape[1])])
        titlestr = 'Number'
        if(ratio == True):
            masked_pixels = masked_pixels / \
                (self._wfs.shape[2] * self._wfs.shape[3])
            titlestr = 'Fraction'
        plt.figure()
        plt.clf()
        plt.ion()
        plt.plot(masked_pixels)

        plt.ylabel(titlestr + ' of Masked Pixels', size=25)
        plt.xlabel('Measures', size=25)
        plt.title('Number of scans per actuator:%d' %
                  self._wfs.shape[1])

    def add_repeated_measure(self, cplm_to_add, act_list):
        for idx, act in enumerate(act_list):
            self._wfs[act] = cplm_to_add._wfs[idx]

    def save_results(self, fname):
        hdr = fits.Header()
        hdr['REF_TAG'] = self._reference_tag
        hdr['N_AV_FR'] = self.NUMBER_WAVEFRONTS_TO_AVERAGE
        fits.writeto(fname, self._wfs.data, hdr)
        fits.append(fname, self._wfs.mask.astype(int))
        fits.append(fname, self._cmd_vector)
        fits.append(fname, self._actuators_list)
        fits.append(fname, self._reference_cmds)

    @staticmethod
    def load(fname):
        header = fits.getheader(fname)
        hduList = fits.open(fname)
        wfs_data = hduList[0].data
        wfs_mask = hduList[1].data.astype(bool)
        wfs = np.ma.masked_array(data=wfs_data, mask=wfs_mask)
        cmd_vector = hduList[2].data
        actuators_list = hduList[3].data
        reference_commands = hduList[4].data
        return {'wfs': wfs,
                'cmd_vector': cmd_vector,
                'actuators_list': actuators_list,
                'reference_shape': reference_commands,
                'reference_shape_tag': header['REF_TAG']
                }


class PixelRuler():
    # Mems info
    DefaultPitchInMeters = 450.e-6
    DefaultApertureInMeters = 4.95e-3
    DistanceInPitch = 7
    NactInDistanceInPitch = DistanceInPitch + 1
    NactsInLine = 12
    ActsAroundBarycenter = [63, 64, 75, 76]

    def __init__(self, cpla):
        self._cpla = cpla
        self._n_of_act = self._cpla._wfs.shape[0]

    def _get_pixel_distance_between_peaks(self, act1, act2):
        y1, x1 = self._cpla._get_max_pixel(act1)
        y2, x2 = self._cpla._get_max_pixel(act2)
        return np.sqrt((y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1))

    def _get_pixel_distance_alongX(self):
        act1 = np.arange(24, 31 + 1)
        act2 = np.arange(108, 115 + 1)
        pixdist = np.zeros(self.NactInDistanceInPitch)
        for i in range(self.NactInDistanceInPitch):
            pixdist[i] = self._get_pixel_distance_between_peaks(
                act1[i], act2[i])
        return pixdist

    def _get_pixel_distance_alongY(self):
        act1 = np.arange(31, 127, self.NactsInLine)
        act2 = np.arange(24, 120, self.NactsInLine)
        pixdist = np.zeros(self.NactInDistanceInPitch)
        for i in range(self.NactInDistanceInPitch):
            pixdist[i] = self._get_pixel_distance_between_peaks(
                act1[i], act2[i])
        return pixdist

    def get_barycenter_around_actuators(self):
        y = np.zeros_like(self.ActsAroundBarycenter)
        x = np.zeros_like(self.ActsAroundBarycenter)
        for i, act in enumerate(self.ActsAroundBarycenter):
            y[i], x[i] = self._cpla._get_max_pixel(act)
        y_barycenter = y.sum() // len(y)
        x_barycenter = x.sum() // len(x)
        return y_barycenter, x_barycenter

    def build_pixel_dim_in_meters(self):
        distance_list = []
        distance = self._get_pixel_distance_alongX()
        for dist in distance:
            distance_list.append(dist)
        distance = self._get_pixel_distance_alongY()
        for dist in distance:
            distance_list.append(dist)
        distance_list.append(
            self._get_pixel_distance_between_peaks(31, 108) / np.sqrt(2))
        distance_list.append(
            self._get_pixel_distance_between_peaks(24, 115) / np.sqrt(2))
        distance_list = np.array(distance_list)
        self._pixel_mean_dimension_in_meters = self.DistanceInPitch * \
            self.DefaultPitchInMeters / distance_list.mean()
        self._pixel_sigma_dimension_in_meters = self._pixel_mean_dimension_in_meters * \
            distance_list.std() / distance_list.mean()

    def pixel2meter(self, LenghtInPixels):
        return LenghtInPixels * self._pixel_mean_dimension_in_meters

    def meter2pixel(self, LenghtInMeters):
        return LenghtInMeters / self._pixel_mean_dimension_in_meters
