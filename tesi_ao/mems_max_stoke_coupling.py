import numpy as np
from tesi_ao.main220316 import create_devices
from tesi_ao.mems_command_linearization import MemsCommandLinearization
from astropy.io import fits


class CouplingMeasurer():

    def __init__(self, upper_left_corner_act, n, mcl_fname=None):
        self._wyko, self._bmc = create_devices()
        if mcl_fname is None:
            mcl_fname = 'prova/misure_ripetute/mcl_all_def.fits'
        self._mcl = MemsCommandLinearization.load(mcl_fname)
        self.act_list = self.get_nxn_act_list(upper_left_corner_act, n)

    def execute_max_stroke_measure_in_nxn(self):
        #self.central_act = act
        #self.act_list = self.get_nxn_act_list(act, 3)
        num_of_selected_act = len(self.act_list)
        cmd0 = np.zeros(self.get_num_of_acts)
        cmd = np.zeros(self.get_num_of_acts)

        self._bmc.set_shape(cmd0)
        wf_flat = self.get_wavefront

        cmd[self.act_list] = self._mcl._cmd_vector[self.act_list, 0]
        self._bmc.set_shape(cmd)
        wf = self.get_wavefront
        wf = wf - wf_flat
        self.wf_pos = wf - np.ma.median(wf)

        self._bmc.set_shape(cmd0)
        wf_flat = self.get_wavefront

        cmd[self.act_list] = self._mcl._cmd_vector[self.act_list, -1]
        self._bmc.set_shape(cmd)
        wf = self.get_wavefront
        wf = wf - wf_flat
        self.wf_neg = wf - np.ma.median(wf)

        self._bmc.set_shape(cmd0)
        return self.wf_pos, self.wf_neg

    def get_nxn_act_list(self, upper_left_corner_act, n):
        up = upper_left_corner_act
        n_of_act = n * n
        act_list = np.zeros(n_of_act, dtype=int)
        for i in range(0, n):
            a = n * i
            b = a + n
            act_list[a:b] = np.arange(
                up - (n - 1), up + 1, dtype=int) + 12 * int(i)
        return act_list

    def poke_nxn_actuators(self, pos):
        self.pos = pos
        #self.act_list = self.get_nxn_act_list(upper_left_corner_act, n)
        num_of_selected_acts = len(self.act_list)
        cmd0 = np.zeros(self.get_num_of_acts)
        cmd = np.zeros(self.get_num_of_acts)
        frame_shape = self.get_frame_shape
        self.wfs_poke = np.ma.zeros(
            (num_of_selected_acts, frame_shape[0], frame_shape[1]))
        self.act_pix_coord = np.zeros((num_of_selected_acts, 2), dtype=int)

        for idx, act in enumerate(self.act_list):
            self._bmc.set_shape(cmd0)
            wf_flat = self.get_wavefront
            cmd[act] = self._mcl._sampled_p2c(act, pos)
            self._bmc.set_shape(cmd)
            wf = self.get_wavefront
            wf_sub = wf - wf_flat
            self.wfs_poke[idx] = wf_sub - np.ma.median(wf_sub)
            self.act_pix_coord[idx] = self._get_act_stroke_coord_from_wf(
                self.wfs_poke[idx])
            cmd[act] = 0.

    def _get_act_stroke_coord_from_wf(self, wf):
        coord_max = np.argwhere(np.abs(wf) == np.max(np.abs(wf)))[0]
        y, x = coord_max[0], coord_max[1]
        return y, x

    def get_act_stroke_from_wf(self, y, x, wf):

       # y, x = self._get_act_stroke_coord_from_wf(wf)
        list_to_avarage = []
        # avoid masked data
        for yi in range(y - 1, y + 2):
            for xi in range(x - 1, x + 2):
                if(wf[yi, xi].data != 0.):
                    list_to_avarage.append(wf[yi, xi])
        list_to_avarage = np.array(list_to_avarage)
        # return wf[y, x]
        return np.median(list_to_avarage)

    def get_actuators_ptv(self):
        pos140 = np.zeros(self.get_num_of_acts)
        for idx, act in enumerate(self.act_list):
            y, x = self.act_pix_coord[idx]
            #peak = self.wf_pos[y, x]
            #valley = self.wf_neg[y, x]
            peak = self.get_act_stroke_from_wf(y, x, self.wf_pos)
            valley = self.get_act_stroke_from_wf(y, x, self.wf_neg)
            pos140[int(act)] = peak - valley
        return pos140

    def get_poke_act_stoke(self):
        pos140 = np.zeros(self.get_num_of_acts)
        n_sel_act = len(self.act_list)
        middle_idx = int(n_sel_act * 0.5)
        wf = self.wfs_poke[middle_idx]
        for idx, act in enumerate(self.act_list):
            y, x = self.act_pix_coord[idx]
            list_to_avarage = []
            # avoid masked data
            for yi in range(y - 1, y + 2):
                for xi in range(x - 1, x + 2):
                    if(wf[yi, xi].data != 0.):
                        list_to_avarage.append(wf[yi, xi])
            list_to_avarage = np.array(list_to_avarage)
            pos140[act] = np.median(list_to_avarage)
        return pos140

    @property
    def get_num_of_acts(self):
        return self._bmc.get_number_of_actuators()

    @property
    def get_wavefront(self):
        return self._wyko.wavefront(timeout_in_sec=10)

    @property
    def get_frame_shape(self):
        wf_temp = self.get_wavefront
        return wf_temp.shape

    def save_results(self, fname):
        hdr = fits.Header()
        hdr['POS'] = self.pos
        fits.writeto(fname, self.act_list, hdr)
        fits.append(fname, self.wfs_poke.data)
        fits.append(fname, self.wfs_poke.mask.astype(int))
        fits.append(fname, self.wf_pos.data)
        fits.append(fname, self.wf_pos.mask.astype(int))
        fits.append(fname, self.wf_neg.data)
        fits.append(fname, self.wf_neg.mask.astype(int))
        fits.append(fname, self.act_pix_coord)

    @staticmethod
    def load(fname):
        header = fits.getheader(fname)
        hduList = fits.open(fname)
        pos = header['POS']
        selected_acts = hduList[0].data
        wfs_data = hduList[1].data
        wfs_mask = hduList[2].data.astype(bool)
        wfs_poke = np.ma.array(data=wfs_data, mask=wfs_mask)
        wfs_data = hduList[3].data
        wfs_mask = hduList[4].data.astype(bool)
        wf_pos = np.ma.array(data=wfs_data, mask=wfs_mask)
        wfs_data = hduList[5].data
        wfs_mask = hduList[6].data.astype(bool)
        wf_neg = np.ma.array(data=wfs_data, mask=wfs_mask)
        act_pix_coord = hduList[7].data

        return pos, selected_acts, wfs_poke, wf_pos, wf_neg, act_pix_coord
