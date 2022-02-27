
import numpy as np
from plico_interferometer import interferometer
from plico_dm import deformableMirror
from astropy.io import fits
from scipy.interpolate.interpolate import interp1d


def create_devices():
    wyko = interferometer('193.206.155.29', 7300)
    bmc = deformableMirror('193.206.155.92', 7000)
    return wyko, bmc


class CommandToPositionLinearizationMeasurer(object):

    NUMBER_WAVEFRONTS_TO_AVERAGE = 3
    NUMBER_STEPS_VOLTAGE_SCAN = 11

    def __init__(self, interferometer, mems_deformable_mirror):
        self._interf = interferometer
        self._bmc = mems_deformable_mirror
        self._n_acts = self._bmc.get_number_of_modes()
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

    def _get_wavefront_flat_subtracted(self):
        dd = self._interf.wavefront(
            self.NUMBER_WAVEFRONTS_TO_AVERAGE) - self._get_zero_command_wavefront()
        return dd - np.ma.median(dd)

    def _reset_flat_wavefront(self):
        self._wfflat = None

    def save_results(self, fname):
        hdr = fits.Header()
        hdr['REF_TAG'] = self._reference_tag
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

    def _max_wavefront(self, act, cmd_index):
        wf = self._wfs[act, cmd_index]
        coord_max = np.argwhere(np.abs(wf) == np.max(np.abs(wf)))[0]
        return wf[coord_max[0], coord_max[1]]

    def _max_roi_wavefront(self, act, cmd_index):
        wf = self._wfs[act, cmd_index]
        b, t, l, r = self._get_max_roi(act)
        wfroi = wf[b:t, l:r]
        coord_max = np.argwhere(
            np.abs(wfroi) == np.max(np.abs(wfroi)))[0]
        return wfroi[coord_max[0], coord_max[1]]

    def _get_max_roi(self, act):
        roi_size = 50
        wf = self._wfs[act, 0]
        coord_max = np.argwhere(np.abs(wf) == np.max(np.abs(wf)))[0]
        return coord_max[0] - roi_size, coord_max[0] + roi_size, \
            coord_max[1] - roi_size, coord_max[1] + roi_size

    def _max_vector(self, act):
        res = np.zeros(self._n_steps_voltage_scan)
        for i in range(self._n_steps_voltage_scan):
            res[i] = self._max_roi_wavefront(act, i)
        return res

    def _compute_maximum_deflection(self):
        self._max_deflection = np.array([
            self._max_vector(i) for i in range(len(self._actuators_list))])

    def compute_linearization(self):
        self._compute_maximum_deflection()

        return MemsCommandLinearization(
            self._actuators_list,
            self._cmd_vector,
            self._max_deflection,
            self._reference_shape_tag)


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


class Robaccia220223(object):

    def __init__(self, cmd_vector, amplitude_vector):
        x = cmd_vector
        y = amplitude_vector
        res = np.polyfit(x, y, 2)
        self.a = res[0]
        self.b = res[1]
        self.c = res[2]

    def quadratic_fit(self):
        n_acts_to_meas = len(self._actuators_list)
        self._quadratic_coeffs = np.zeros((n_acts_to_meas, 3))

        for index in range(n_acts_to_meas):
            x = self._cmd_vector[index]
            y = self._max_vector(index)
            res = np.polyfit(x, y, 2)
            self._quadratic_coeffs[index] = res

    def _get_quadratic_coeffs(self, act):
        actidx = np.argwhere(self._actuators_list == act)[0][0]
        a = self._quadratic_coeffs[actidx, 0]
        b = self._quadratic_coeffs[actidx, 1]
        c = self._quadratic_coeffs[actidx, 2]
        return a, b, c

    def c2p(self, act, v):
        a, b, c = self._get_quadratic_coeffs(act)
        return a * v**2 + b * v + c

    def p2c(self, act, p):
        a, b, c = self._get_quadratic_coeffs(act)
        v = (-b - np.sqrt(b**2 - 4 * a * (c - p))) / (2 * a)
        return v
