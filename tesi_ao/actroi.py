import numpy as np
from plico_interferometer import interferometer
from plico_dm import deformableMirror
from astropy.io import fits
import random
from scipy.optimize import curve_fit


def create_devices():
    wyko = interferometer('193.206.155.29', 7300)
    bmc = deformableMirror('193.206.155.92', 7000)
    return wyko, bmc

# Trying to spot the pixel area relative to the actuator
# giving a one by one unit command
# trying to execute a 2Dgaussian fit on peaks and save the relative amplitudes
# pixel_area =?= sigmax*sigmay (estimated from gaussian fitting)


class ActuatorsRegionMeasurer(object):

    NUMBER_WAVEFRONTS_TO_AVERAGE = 1
    NUMBER_STEPS_VOLTAGE_SCAN = 3

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

        for act_idx, act in enumerate(self._actuators_list):
            self._cmd_vector[act_idx] = np.ones(
                self.NUMBER_STEPS_VOLTAGE_SCAN) - self._reference_cmds[act]
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


class ActuatorsRegionAnalyzer(object):

    def __init__(self, scan_fname):
        res = ActuatorsRegionMeasurer.load(scan_fname)
        self._wfs = res['wfs']
        self._cmd_vector = res['cmd_vector']
        self._actuators_list = res['actuators_list']
        self._reference_shape_tag = res['reference_shape_tag']
        self._n_steps_voltage_scan = self._wfs.shape[1]

    def _max_roi_wavefront(self, act_idx, cmd_index):
        wf = self._wfs[act_idx, cmd_index]
        b, t, l, r = self._get_max_roi(act_idx)
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

    def _max_vector(self, act_idx):
        res = np.zeros(self._n_steps_voltage_scan)
        for i in range(self._n_steps_voltage_scan):
            res[i] = self._max_roi_wavefront(act_idx, i)
        return res

    def _compute_maximum_deflection(self):
        self._max_deflection = np.array([
            self._max_vector(act_idx) for act_idx in range(len(self._actuators_list))])

    def _2dgaussian(self, X, amplitude, x0, y0, sigmax, sigmay, offset):
        x, y = X
        a = 0.5 * ((x - x0) / sigmax)**2
        b = 0.5 * ((y - y0) / sigmay)**2
        z = amplitude * np.exp(-(a + b)) + offset
        return z

    def _gaussian_fitting(self, act_idx, cmd_index):
        wf = self._wfs[act_idx, cmd_index]
        b, t, l, r = self._get_max_roi(act_idx)
        wfroi = wf[b:t, l:r]
        z = wfroi
        x = np.arange(wfroi.shape[1])
        y = np.arange(wfroi.shape[0])

        A0 = self._max_roi_wavefront(act_idx, cmd_index)
        x0 = 49.
        y0 = 49.
        sigma0 = z.std()
        offset = 0
        starting_values = [A0, x0, y0, sigma0, sigma0, offset]
        #error: 'result not a proper array of floats...due masked array?'
        ppar, pcov = curve_fit(self._2dgaussian, (x, y), z, p0 =starting_values)) 
        return ppar, pcov