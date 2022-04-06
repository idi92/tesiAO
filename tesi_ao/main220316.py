
import numpy as np
from plico_interferometer import interferometer
from plico_dm import deformableMirror
from astropy.io import fits
from scipy.interpolate.interpolate import interp1d
from functools import reduce
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy import dtype
from arte.types.mask import CircularMask
from arte.utils.zernike_generator import ZernikeGenerator


def create_devices():
    wyko = interferometer('193.206.155.29', 7300)
    bmc = deformableMirror('193.206.155.92', 7000)
    return wyko, bmc


class CommandToPositionLinearizationMeasurer(object):

    NUMBER_WAVEFRONTS_TO_AVERAGE = 1
    NUMBER_STEPS_VOLTAGE_SCAN = 11

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
                self.NUMBER_WAVEFRONTS_TO_AVERAGE)
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
            self.NUMBER_WAVEFRONTS_TO_AVERAGE) - self._get_zero_command_wavefront()
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
        self._create_interpolation()

    def _create_interpolation(self):
        self._finter = [interp1d(
            self._deflection[i], self._cmd_vector[i], kind='cubic')
            for i in range(self._cmd_vector.shape[0])]

    def _get_act_idx(self, act):
        return np.argwhere(self._actuators_list == act)[0][0]

    def p2c(self, act, p):
        idx = self._get_act_idx(act)
        return self._finter[idx](p)

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


class PupilMaskBuilder():

    def __init__(self, wfmask):
        self._wfmask = wfmask

    def get_circular_mask(self, radius, center):
        mask = CircularMask(self._wfmask.shape,
                            maskRadius=radius, maskCenter=center)
        return mask.mask()

    def get_centered_circular_mask(self):
        '''
        centered wrt the intersection mask
        '''
        circular_mask = self._wfmask
        # center pixel coord of the full map
        central_ypix = self._wfmask.shape[0] // 2
        central_xpix = self._wfmask.shape[1] // 2
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

    NORM_AT_THIS_CMD = 13  # such that wyko noise and saturation are avoided
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

    # def create_circular_mask(self, radius, center):
    #     mask = CircularMask(self._imask.shape,
    #                         maskRadius=radius, maskCenter=center)
    #     return mask.mask()
    #
    # def create_centered_circular_mask(self):
    #     '''
    #     centered wrt the intersection mask
    #     '''
    #     circular_imask = self._imask
    #     # center pixel coord of the full map
    #     central_ypix = self._imask.shape[0] // 2
    #     central_xpix = self._imask.shape[1] // 2
    #     HeightInPixels = (self._imask[:, central_xpix] == False).sum()
    #     WidthInPixels = (self._imask[central_ypix, :] == False).sum()
    #
    #     offsetX = (self._imask[central_ypix,
    #                            0:self._imask.shape[1] // 2] == True).sum()
    #     offsetY = (self._imask[0:self._imask.shape[0] //
    #                            2, central_xpix] == True).sum()
    #     # center of False map and origin of circular mask in pixel
    #     yc0 = offsetY + HeightInPixels // 2
    #     xc0 = offsetX + WidthInPixels // 2
    #     RadiusInPixels = min(WidthInPixels, HeightInPixels) // 2
    #     for j in range(self._imask.shape[0]):
    #         for i in range(self._imask.shape[1]):
    #             Distanceinpixels = np.sqrt(
    #                 (j - yc0)**2 + (i - xc0)**2)
    #             if(Distanceinpixels <= RadiusInPixels):
    #                 circular_imask[j, i] = False
    #             else:
    #                 circular_imask[j, i] = True
    #     return circular_imask

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

    def compute_reconstructor(self, mask=None):
        # TODO: check that mask.shape is equal to self._imask.shape
        if mask is None:
            mask = self._imask
        assert self._imask.shape == mask.shape, f"mask has not the same dimension of self._imask!\nGot:{mask.shape}\nShould be:{self._imask.shape}"
        self._pupil_mask = np.ma.mask_or(self._imask, mask)
        self._build_valid_actuators_list()
        self._build_interaction_matrix()
        self._build_reconstruction_matrix()

    def generate_mode(self, wfmap):
        self._wfmode = np.ma.array(data=wfmap, mask=self._pupil_mask)

    def generate_zernike_mode_on_pupil(self, j, diameter, AmpInMeters):
        # PixelsInPupil = (self._pupil_mask == False).sum()
        # PupilDiameterInPixels = np.sqrt(4 * PixelsInPupil / np.pi))
        # TODO: compute diameter from pupil_mask
        # TODO: check zernike_generator:
        #       find a way to input pupil_mask
        PupilDiameterInPixels = diameter
        zg = ZernikeGenerator(PupilDiameterInPixels)
        self._wfmode = np.zeros(self._pupil_mask.shape)
        self._wfmode = np.ma.array(data=self._wfmode, mask=self._pupil_mask)
        z_mode = zg.getZernike(j)
        a = (z_mode.mask == False).sum()
        b = (self._wfmode.mask == False).sum()
        assert a == b, f"zerike valid points: {a}  wfmode valid points: {b}\nShould be equal!"
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
        amp = self._wfmode.std()
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

    def __init__(self, interferometer, mems_deformable_mirror):
        self._interf = interferometer
        self._bmc = mems_deformable_mirror

    def execute_measure(self, mcl, mg, pos=None):
        if pos is None:
            pos = mg.get_position_cmds_from_wf()

        flat_cmd = np.zeros(self._bmc.get_number_of_actuators())
        self._bmc.set_shape(flat_cmd)

        expected_valid_points = (mg._pupil_mask == False).sum()

        wfflat = self._interf.wavefront()
        wfflat = np.ma.array(data=wfflat, mask=mg._pupil_mask)
        # avoid nasty wf maps
        measured_valid_points = (wfflat.mask == False).sum()
        ratio = measured_valid_points / expected_valid_points
        while(ratio < self.THRESHOLD_RATIO):
            print('Warning: Nasty map acquired!Reloading...')
            wfflat = self._interf.wavefront()
            wfflat = np.ma.array(data=wfflat, mask=mg._pupil_mask)
            measured_valid_points = (wfflat.mask == False).sum()
            ratio = measured_valid_points / expected_valid_points

        act_list = mg._acts_in_pupil
        cmd = np.zeros(self._bmc.get_number_of_actuators())
        # TODO: clip voltage!
        for idx in range(len(pos)):
            cmd[act_list[idx]] = mcl.p2c(act_list[idx], pos[idx])
        self._bmc.set_shape(cmd)
        #_get_wavefront_flat_subtracted
        wfflatsub = self._interf.wavefront() - wfflat
        self._wfmeas = wfflatsub - np.ma.median(wfflatsub)
        self._wfmeas = np.ma.array(data=self._wfmeas, mask=mg._pupil_mask)
        # avoid nasty wf maps
        measured_valid_points = (self._wfmeas.mask == False).sum()
        ratio = measured_valid_points / expected_valid_points
        while(ratio < self.THRESHOLD_RATIO):
            print('Warning: Nasty map acquired!Reloading...')
            wfflatsub = self._interf.wavefront() - wfflat
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

      
class ActuatorsInPupilThresholdAnalyzer():
    #TODO: data la pupilla e dati i modi che si vogliono generare,
    # determinare la lista degli attuatori che minimizzano il fitting error osservato
    # sia per ciascun modo che per quelli che si vogliono generare
    # to be continued...
    Nmeasures = 20
    THRESHOLD_SPAN = np.linspace(0.01, 0.5, Nmeasures)

    def __init__(self, mg, mm, pupil_mask):
        self._mode_generator = mg
        self._mode_measurer = mm
        self._pupil_mask = pupil_mask
    
    def _spot_threshold_per_mode(self, j):
        for threshold in self.THRESHOLD_SPAN:
        self._mode_generator.THRESHOLD_RMS=threshold
        self._mode_generator.compute_reconstructor(mask=self._pupil_mask)
        self._mode_generator.generate_zernike_mode_on_pupil(j,240,50.e-9)
        self._mode_generator.build_fitted_wavefront()
        amp = self._mode_generator._wfmode.std()
        fitting_error = (self._mode_generator._wffitted - self._mode_generator._wfmode).std()
        
        


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


class InfluenceFunctionMeasurer():
    NUMBER_STEPS_VOLTAGE_SCAN = 1
    NUMBER_WAVEFRONTS_TO_AVERAGE = 1

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
                self.NUMBER_WAVEFRONTS_TO_AVERAGE)
        return self._wfflat

    def PrintCommonUnitCommandInterval(self, mcl):
        MaxUnitCommand = np.min(mcl._deflection[:, 0])
        MinUnitCommand = np.max(mcl._deflection[:, -1])
        print(
            'Select Unit deflection in the following interval:[%g,' % MinUnitCommand + ' %g] Meters' % MaxUnitCommand)

    def execute_unit_command_scan(self, mcl, UnitCmdInMeters=None, act_list=None):
        if UnitCmdInMeters is None:
            UnitCmdInMeters = 200.e-9
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
            self._cmd_vector[act_idx] = mcl.p2c(act, UnitCmdInMeters)
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
            self.NUMBER_WAVEFRONTS_TO_AVERAGE) - self._get_zero_command_wavefront()
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

    def get_pixel_distance_alongX(self):
        act1 = np.arange(24, 31 + 1)
        act2 = np.arange(108, 115 + 1)
        pixdist = np.zeros(self.NactInDistanceInPitch)
        for i in range(self.NactInDistanceInPitch):
            pixdist[i] = self._get_pixel_distance_between_peaks(
                act1[i], act2[i])
        return pixdist

    def get_pixel_distance_alongY(self):
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
