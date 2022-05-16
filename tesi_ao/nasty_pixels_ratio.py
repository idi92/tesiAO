
import numpy as np
from plico_interferometer import interferometer
from plico_dm import deformableMirror
from astropy.io import fits

import matplotlib.pyplot as plt


def create_devices():
    wyko = interferometer('193.206.155.29', 7300)
    bmc = deformableMirror('193.206.155.92', 7000)
    return wyko, bmc


class ReasonableMaskedPixelsMeasurer(object):

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

    def execute_flat_measure(self, num_of_meas=10):
        self._n_meas = num_of_meas
        wfs_meas = []
        for i in np.arange(num_of_meas):

            cmd = np.zeros(self._bmc.get_number_of_actuators())
            self._bmc.set_shape(cmd)
            wf = self._interf.wavefront(
                self.NUMBER_WAVEFRONTS_TO_AVERAGE, timeout_in_sec=self.TIME_OUT)
            wfs_meas.append(wf)
            print('Measure %d acquired!' % i)
        self._measured_flat_maps = np.ma.array(wfs_meas)

    def execute_command_scan(self, act_list=None):
        # the idea is to do a scan in cmds for each
        # act without any ctrl on nasty pixels
        # in order to see the mask coverage pattern
        # while acquiring measures
        # just setting stuff on 4sight
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

        #N_pixels = self._wfs.shape[2] * self._wfs.shape[3]
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
                # masked_pixels = self._wfs[act_idx, cmd_idx].mask.sum()
                # masked_ratio = masked_pixels / N_pixels
                # if masked_ratio > 0.7829:
                #     print('Warning: Bad measure acquired for: act%d' %
                #           act_idx + ' cmd_idx %d' % cmd_idx)
                #     self._avoid_saturated_measures(
                #         masked_ratio, act_idx, cmd_idx, N_pixels)
            self.reset_flat_wavefront()

    # def _avoid_saturated_measures(self, masked_ratio, act_idx, cmd_idx, N_pixels):
    #
    #     while masked_ratio > 0.7829:
    #         self._wfs[act_idx, cmd_idx, :,
    #                   :] = self._get_wavefront_flat_subtracted()
    #         masked_pixels = self._wfs[act_idx, cmd_idx].mask.sum()
    #         masked_ratio = masked_pixels / N_pixels
    #
    #     print('Repeated measure completed!')

    def _get_wavefront_flat_subtracted(self):
        dd = self._interf.wavefront(
            self.NUMBER_WAVEFRONTS_TO_AVERAGE, timeout_in_sec=self.TIME_OUT) - self._get_zero_command_wavefront()
        return dd - np.ma.median(dd)

    def get_reasonable_masked_pixel_ratio(self):
        num_masked_pixels = np.zeros(self._n_meas)
        Npixel_frame = self._measured_flat_maps.shape[1] * \
            self._measured_flat_maps.shape[2]
        for i in np.arange(self._n_meas):
            num_masked_pixels[i] = self._measured_flat_maps[i].mask.sum()
        masked_pixel_ratio = num_masked_pixels / Npixel_frame
        return masked_pixel_ratio

    def reset_flat_wavefront(self):
        self._wfflat = None

    def display_mask_coverage(self, ratio=False):
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
