import numpy as np
from plico_interferometer import interferometer
from plico_dm import deformableMirror
from astropy.io import fits
import random


def _what_Ive_done():
    wyko, bmc = create_devices()
    # static measure of Mems reference shape
    fm100st = Flat_Measurer(wyko, bmc)
    fm100st.NUMBER_STEPS_VOLTAGE_SCAN = 100  # repeated measures
    fm100st.execute_static_scan([63])
    fm100st.save_results('prova/act63/flatstab/fm100st.fits')

    fa100st = Flat_Analyzer('prova/act63/flatstab/fm100st.fits')

    rms100st = fa100st.get_rms_in_each_pixel()
    plt.figure(123)
    plt.clf()
    # par_map[act, yi, xi] == rms_map[act, yi, xi]
    plt.imshow(rms100st.par_map[0])
    plt.colorbar()

    rms100st.save_results('prova/act63/flatstab/rms100st.fits')
    rms100st.load('prova/act63/flatstab/rms100st.fits')
    # turning up and down repeated measures
    fm100ud = Flat_Measurer(wyko, bmc)
    fm100ud.NUMBER_STEPS_VOLTAGE_SCAN = 100  # repeated measures
    fm100ud.execute_up_and_down_scan([63])
    fm100ud.save_results('prova/act63/flatstab/fm100ud.fits')

    fa100ud = Flat_Analyzer('prova/act63/flatstab/fm100ud.fits')

    rms100ud = fa100ud.get_rms_in_each_pixel()
    plt.figure(124)
    plt.clf()
    plt.imshow(rms100ud.par_map[0])
    plt.colorbar()

    rms100ud.save_results('prova/act63/flatstab/rms100st.fits')
    rms100ud.load('prova/act63/flatstab/rms100st.fits')


def create_devices():
    wyko = interferometer('193.206.155.29', 7300)
    bmc = deformableMirror('193.206.155.92', 7000)
    return wyko, bmc

# Studying Mems flat shape stability
# in order to understand rms around 0nm
# in x_obs - x_exp vs x_exp plot


class Flat_Measurer(object):

    NUMBER_WAVEFRONTS_TO_AVERAGE = 1
    NUMBER_STEPS_VOLTAGE_SCAN = 10

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

    def execute_static_scan(self, act_list=None):
        '''
        Acquires WF maps while the DM is fixed to its reference shape
        '''
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
            self._cmd_vector[act_idx] = - \
                self._reference_cmds[act] + self._reference_cmds[act]
            for cmd_idx, cmdi in enumerate(self._cmd_vector[act_idx]):
                print("Act:%d - command %g" % (act, cmdi))
                cmd = np.zeros(self._n_acts)
                cmd[act] = cmdi
                self._bmc.set_shape(cmd)
                self._wfs[act_idx, cmd_idx, :,
                          :] = self._get_wavefront_flat_subtracted()

    def execute_up_and_down_scan(self, act_list=None):
        '''
        Acquires flat WF maps, after a random command given to the actuator
        '''
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
            self._cmd_vector[act_idx] = - \
                self._reference_cmds[act] + self._reference_cmds[act]
            for cmd_idx, cmdi in enumerate(self._cmd_vector[act_idx]):
                print("Act:%d - command %g" % (act, cmdi))
                cmd = np.zeros(self._n_acts)
                cmd[act] = random.uniform(0, 1) - self._reference_cmds[act]
                self._bmc.set_shape(cmd)
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


class Flat_Analyzer(object):

    def __init__(self, scan_fname):
        res = Flat_Measurer.load(scan_fname)
        self._wfs = res['wfs']
        self._cmd_vector = res['cmd_vector']
        self._actuators_list = res['actuators_list']
        self._reference_shape_tag = res['reference_shape_tag']
        self._num_of_measures = self._wfs.shape[1]

    def get_rms_in_each_pixel(self):
        '''
        Estimates std.dev in each measured maps pixel
        '''
        rms_map = np.ma.empty_like(self._wfs.data[:, 0])

        for act in np.arange(self._wfs.shape[0]):
            for yi in np.arange(self._wfs.shape[2]):
                for xi in np.arange(self._wfs.shape[3]):
                    rms_map[act, yi, xi] = self._wfs[act, :, yi, xi].std()

        return CollapsedMap(rms_map, self._actuators_list, self._num_of_measures)

    def get_mean_in_each_pixel(self):

        mean_map = np.ma.empty_like(self._wfs.data[:, 0])

        for act in np.arange(self._wfs.shape[0]):
            for yi in np.arange(self._wfs.shape[2]):
                for xi in np.arange(self._wfs.shape[3]):
                    mean_map[act, yi, xi] = self._wfs[act, :, yi, xi].mean()

        return CollapsedMap(mean_map, self._actuators_list, self._num_of_measures)


class CollapsedMap(object):

    def __init__(self, par_map, act_list, num_of_meas):
        self.par_map = par_map
        self.act_list = act_list
        self.num_of_meas = num_of_meas

    def save_results(self, fname):
        hdr = fits.Header()
        hdr['N_Meas'] = self.num_of_meas
        fits.writeto(fname, self.par_map.data, hdr)
        fits.append(fname, self.par_map.mask.astype(int))
        fits.append(fname, self.act_list)

    @staticmethod
    def load(fname):
        header = fits.getheader(fname)
        hduList = fits.open(fname)
        par_data = hduList[0].data
        par_mask = hduList[1].data.astype(bool)
        par_map = np.ma.masked_array(data=par_data, mask=par_mask)
        actuators_list = hduList[2].data
        num_of_meas = header['N_Meas']

        return CollapsedMap(par_map, actuators_list, num_of_meas)

    '''
    #I want to estimate the rms in each actuator's pixel area
    #while MEMs has a flat shape (trying to save them into a file
    #executed by actroi.py)
  
    '''
