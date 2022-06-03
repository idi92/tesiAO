import numpy as np
from astropy.io import fits
from tesi_ao.mems_command_linearization import MemsCommandLinearization
from functools import reduce


class ZonalInfluenceFunctionMeasurer(object):

    TIMEOUT = 10

    def __init__(self, interferometer, mems_deformable_mirror, mcl_fname=None):
        self._interf = interferometer
        self._bmc = mems_deformable_mirror
        self._num_of_acts = self._bmc.get_number_of_actuators()
        if mcl_fname is None:
            mcl_fname = 'prova/all_act/sandbox/mcl_all_fixedpix.fits'
        self._mcl = MemsCommandLinearization.load(mcl_fname)

    def execute_ifs_measure(self, pos, actuators_list=None):
        if actuators_list is None:
            self.actuators_list = np.arange(self._num_of_acts)
        else:
            self.actuators_list = np.array(actuators_list)

        cmd = np.zeros(self._num_of_acts)
        self._bmc.set_shape(cmd)

        self._wfflat = self._interf.wavefront(timeout_in_sec=self.TIMEOUT)
        frame_shape = self._wfflat.shape

        self._wfpos = np.ma.zeros(
            (self._num_of_acts, frame_shape[0], frame_shape[1]))
        self._wfneg = np.ma.zeros(
            (self._num_of_acts, frame_shape[0], frame_shape[1]))
        self._wfzero = np.ma.zeros(
            (self._num_of_acts, frame_shape[0], frame_shape[1]))

        self._position_cmd_vector = np.zeros(self._num_of_acts)

        for act in self.actuators_list:
            print('act%d' % act)
            self._position_cmd_vector[act] = 0.
            self._bmc.set_shape(self._mcl.p2c(self._position_cmd_vector))
            self._wfzero[act] = self._interf.wavefront(
                timeout_in_sec=self.TIMEOUT)

            self._position_cmd_vector[act] = pos
            self._bmc.set_shape(self._mcl.p2c(self._position_cmd_vector))
            self._wfpos[act] = self._interf.wavefront(
                timeout_in_sec=self.TIMEOUT)

            self._position_cmd_vector[act] = -pos
            self._bmc.set_shape(self._mcl.p2c(self._position_cmd_vector))
            self._wfneg[act] = self._interf.wavefront(
                timeout_in_sec=self.TIMEOUT)

            self._position_cmd_vector[act] = 0.
        self._pos = pos  # push pull

    def compute_zonal_ifs(self):
        ifs = self._wfpos - self._wfneg
        frame_shape = self._wfpos[0].shape
        dd = np.ma.zeros((self._num_of_acts, frame_shape[0], frame_shape[1]))
        for act in self.actuators_list:
            dd[act] = ifs[act] - np.ma.median(ifs[act])
        self.ifs = 0.5 * dd
        self._apply_intersection_mask()
        return self.ifs

    def _apply_intersection_mask(self):
        imask = reduce(lambda a, b: np.ma.mask_or(
            a, b), self.ifs[:].mask)
        self.ifs[:].mask = imask

    def save_ifs(self, fname):
        hdr = fits.Header()
        hdr['STROKE'] = self._pos
        fits.writeto(fname, self.ifs.data, hdr)
        fits.append(fname, self.ifs.mask.astype(int))
        fits.append(fname, self.actuators_list)

    @staticmethod
    def load_ifs(fname):
        header = fits.getheader(fname)
        hduList = fits.open(fname)
        ifs_data = hduList[0].data
        ifs_mask = hduList[1].data.astype(bool)
        ifs = np.ma.masked_array(data=ifs_data, mask=ifs_mask)
        act_list = hduList[2].data

        stroke = header['STROKE']
        return stroke, act_list, ifs

    def _save_meas(self, fname):
        hdr = fits.Header()
        hdr['STROKE'] = self._pos
        fits.writeto(fname, self._wfpos.data, hdr)
        fits.append(fname, self._wfpos.mask.astype(int))
        fits.append(fname, self._wfzero.data)
        fits.append(fname, self._wfzero.astype(int))
        fits.append(fname, self._wfneg.data)
        fits.append(fname, self._wfneg.astype(int))
        fits.append(fname, self.actuators_list)

    @staticmethod
    def _load_meas(fname):
        header = fits.getheader(fname)
        hduList = fits.open(fname)
        wfpos_data = hduList[0].data
        wfpos_mask = hduList[1].data.astype(bool)
        wfs_pos = np.ma.masked_array(data=wfpos_data, mask=wfpos_mask)

        wfzero_data = hduList[2].data
        wfzero_mask = hduList[3].data.astype(bool)
        wfs_zero = np.ma.masked_array(data=wfzero_data, mask=wfzero_mask)

        wfneg_data = hduList[4].data
        wfneg_mask = hduList[5].data.astype(bool)
        wfs_neg = np.ma.masked_array(data=wfneg_data, mask=wfneg_mask)

        act_list = hduList[6].data

        stroke = header['STROKE']
        return stroke, act_list, wfs_pos, wfs_neg, wfs_zero
