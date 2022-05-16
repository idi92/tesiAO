import numpy as np
from astropy.io import fits

class CommandToPositionLinearizationMeasurer(object):

    NUMBER_WAVEFRONTS_TO_AVERAGE = 1
    NUMBER_STEPS_VOLTAGE_SCAN = 20
    TIME_OUT = 10  # sec
    # use nasty_pixes_ratio.py if u changed the detector
    # mask on 4sight and get a reasonable
    # masked pixel ratio to ctrl and
    # and avoid nasty maps
    # rectangular old mask 0.7829
    # circular new mask
    # 0.8218139146090535
    # 0.821875
    REASONABLE_MASKED_PIXELS_RATIO = 0.8227

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
        self._acquired_wfflat = np.ma.zeros(
            (n_acts_to_meas, wfflat.shape[0], wfflat.shape[1]))

        N_pixels = self._wfs.shape[2] * self._wfs.shape[3]
        for act_idx, act in enumerate(self._actuators_list):
            self._cmd_vector[act_idx] = np.linspace(
                0, 1, self.NUMBER_STEPS_VOLTAGE_SCAN) - self._reference_cmds[act]
            for cmd_idx, cmdi in enumerate(self._cmd_vector[act_idx]):
                print("Act:%d - command %g" % (act, cmdi))
                self._acquired_wfflat[act_idx] = self._get_zero_command_wavefront(
                )
                cmd = np.zeros(self._n_acts)
                cmd[act] = cmdi
                self._bmc.set_shape(cmd)
                self._wfs[act_idx, cmd_idx, :,
                          :] = self._get_wavefront_flat_subtracted()
                masked_pixels = self._wfs[act_idx, cmd_idx].mask.sum()
                masked_ratio = masked_pixels / N_pixels
                if masked_ratio > self.REASONABLE_MASKED_PIXELS_RATIO:
                    print('Warning: Bad measure acquired for: act%d' %
                          act_idx + ' cmd_idx %d' % cmd_idx)
                    self._avoid_saturated_measures(
                        masked_ratio, act_idx, cmd_idx, N_pixels)

            self.reset_flat_wavefront()

    def _avoid_saturated_measures(self, masked_ratio, act_idx, cmd_idx, N_pixels):
        while masked_ratio > self.REASONABLE_MASKED_PIXELS_RATIO:
            self._wfs[act_idx, cmd_idx, :,
                      :] = self._get_wavefront_flat_subtracted()
            masked_pixels = self._wfs[act_idx, cmd_idx].mask.sum()
            masked_ratio = masked_pixels / N_pixels

        print('Repeated measure completed!')

    def _get_wavefront_flat_subtracted(self):
        dd = self._interf.wavefront(
            self.NUMBER_WAVEFRONTS_TO_AVERAGE, timeout_in_sec=self.TIME_OUT) - self._get_zero_command_wavefront()
        return dd - np.ma.median(dd)

    def reset_flat_wavefront(self):
        self._wfflat = None

    def display_mask_coverage(self, ratio=False):
        import matplotlib.pyplot as plt
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
        fits.append(fname, self._acquired_wfflat.data)
        fits.append(fname, self._acquired_wfflat.astype(int))

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
        # TODO: aggiungere try per caricare le misure dei flat
        # dal file nel caso le possieda o meno
        try:
            print('loading...')
            wfs_flat_data = hduList[5].data
            wfs_flat_mask = hduList[6].data.astype(bool)
            wfs_flat = np.ma.masked_array(
                data=wfs_flat_data, mask=wfs_flat_mask)
            return {'wfs': wfs,
                    'cmd_vector': cmd_vector,
                    'actuators_list': actuators_list,
                    'reference_shape': reference_commands,
                    'reference_shape_tag': header['REF_TAG'],
                    'wfs_flat': wfs_flat
                    }

        except IndexError:
            print('In this file: %s' %
                  fname + '\nflat wavefront measurements are missing :( ')
            return {'wfs': wfs,
                    'cmd_vector': cmd_vector,
                    'actuators_list': actuators_list,
                    'reference_shape': reference_commands,
                    'reference_shape_tag': header['REF_TAG']
                    }

