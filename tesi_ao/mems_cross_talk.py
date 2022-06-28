import numpy as np
from tesi_ao.main220316 import create_devices
from tesi_ao.mems_command_linearization import MemsCommandLinearization
from astropy.io import fits


class CrossTalkMeasurer():

    def __init__(self, mcl_fname=None):
        self._wyko, self._bmc = create_devices()
        if mcl_fname is None:
            mcl_fname = 'prova/misure_ripetute/mcl_all_def.fits'
        self._mcl = MemsCommandLinearization.load(mcl_fname)

    def execute_measure_around_actuator(self, act, pos=None):
        if pos is None:
            pos = 1200e-9
        self.pos = pos
        self.act = act
        self.act_list = np.array([act - 12, act - 1, act, act + 1, act + 12])
        num_of_selected_act = len(self.act_list)

        cmd_flat = np.zeros(self.get_num_of_acts)
        cmd = np.zeros(self.get_num_of_acts)
        frame_shape = self.get_frame_shape
        self.wfs = np.ma.zeros(
            (num_of_selected_act, frame_shape[0], frame_shape[1]))

        for idx, act in enumerate(self.act_list):

            self._bmc.set_shape(cmd_flat)
            wf_flat = self.get_wavefront
            cmd[act] = self._mcl._sampled_p2c(act, pos)
            self._bmc.set_shape(cmd)
            wf = self.get_wavefront
            wf_sub = wf - wf_flat
            self.wfs[idx] = wf_sub - np.ma.median(wf_sub)
            cmd[act] = 0.
        self._bmc.set_shape(cmd_flat)

    def save_results(self, fname):
        hdr = fits.Header()
        hdr['ACT'] = self.act
        hdr['POS'] = self.pos
        fits.writeto(fname, self.act_list, hdr)
        fits.append(fname, self.wfs.data)
        fits.append(fname, self.wfs.mask.astype(int))

    @staticmethod
    def load(fname):
        header = fits.getheader(fname)
        hduList = fits.open(fname)
        central_act = header['ACT']
        pos = header['POS']
        selected_acts = hduList[0].data
        wfs_data = hduList[1].data
        wfs_mask = hduList[2].data.astype(bool)
        wfs = np.ma.array(data=wfs_data, mask=wfs_mask)

        return central_act, selected_acts, wfs, pos

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


class CrossTalkAnalyzer():

    def __init__(self, fname):
        self.central_act, self.act_list, self.wfs, self.pos = CrossTalkMeasurer.load(
            fname)

    def get_act_stroke_coord_from_wf(self, wf):
        coord_max = np.argwhere(np.abs(wf) == np.max(np.abs(wf)))[0]
        y, x = coord_max[0], coord_max[1]
        return y, x

    def get_act_stroke_from_wf(self, wf):

        y, x = self.get_act_stroke_coord_from_wf(wf)
        list_to_avarage = []
        # avoid masked data
        for yi in range(y - 1, y + 2):
            for xi in range(x - 1, x + 2):
                if(wf[yi, xi].data != 0.):
                    list_to_avarage.append(wf[yi, xi])
        list_to_avarage = np.array(list_to_avarage)
        # return wf[y, x]
        return np.median(list_to_avarage)

    def show_crosstalk_along_x_axis(self):
        import matplotlib.pyplot as plt
        central_act_idx = np.argwhere(self.central_act == self.act_list)[0][0]
        yc, xc = self.get_act_stroke_coord_from_wf(
            self.wfs[central_act_idx])
        plt.figure()
        plt.clf()
        plt.plot(self.wfs[central_act_idx, yc, :] /
                 1e-6, label='%d' % self.central_act)
        plt.plot(self.wfs[0, yc, :] /
                 1e-6, '--', label='%d' % self.act_list[0])
        plt.plot(self.wfs[-1, yc, :] /
                 1e-6, '--', label='%d' % self.act_list[-1])
        plt.xlabel('pixels along x axis', size=10)
        plt.ylabel('Stroke [$\mu$m]', size=10)

        y_prev, x_prev = self.get_act_stroke_coord_from_wf(
            self.wfs[0])
        vline_max = self.wfs[0, yc, :].max() / 1e-6
        vline_min = self.wfs[0, yc, :].min() / 1e-6
        plt.vlines(x_prev, vline_min, vline_max,
                   colors='r', linestyles='--', linewidth=0.8, alpha=0.5)

        y_foll, x_foll = self.get_act_stroke_coord_from_wf(
            self.wfs[-1])
        vline_max = self.wfs[-1, yc, :].max() / 1e-6
        vline_min = self.wfs[-1, yc, :].min() / 1e-6
        plt.vlines(x_foll, vline_min, vline_max,
                   colors='g', linestyles='--', linewidth=0.8)

        plt.legend(loc='best')
        plt.grid()

    def show_crosstalk_along_y_axis(self):
        import matplotlib.pyplot as plt
        central_act_idx = self.get_central_act_idx
        yc, xc = self.get_act_stroke_coord_from_wf(
            self.wfs[central_act_idx])
        plt.figure()
        plt.clf()
        plt.plot(self.wfs[central_act_idx, :, xc] /
                 1e-6, label='%d' % self.central_act)
        plt.plot(self.wfs[central_act_idx - 1, :, xc] /
                 1e-6, 'r--', label='%d' % self.act_list[central_act_idx - 1])
        plt.plot(self.wfs[central_act_idx + 1, :, xc] /
                 1e-6, 'm--', label='%d' % self.act_list[central_act_idx + 1])
        plt.xlabel('pixels along y axis', size=10)
        plt.ylabel('Stroke [$\mu$m]', size=10)

        y_prev, x_prev = self.get_act_stroke_coord_from_wf(
            self.wfs[central_act_idx + 1])
        vline_max = self.wfs[central_act_idx + 1, :, xc].max() / 1e-6
        vline_min = self.wfs[central_act_idx + 1, :, xc].min() / 1e-6
        plt.vlines(y_prev, vline_min, vline_max,
                   colors='m', linestyles='--', linewidth=0.8)

        y_foll, x_foll = self.get_act_stroke_coord_from_wf(
            self.wfs[central_act_idx - 1])
        vline_max = self.wfs[central_act_idx - 1, :, xc].max() / 1e-6
        vline_min = self.wfs[central_act_idx - 1, :, xc].min() / 1e-6
        plt.vlines(y_foll, vline_min, vline_max,
                   colors='r', linestyles='--', linewidth=0.8)

        plt.legend(loc='best')
        plt.grid()

    def measure_crosstalk(self):
        central_act_idx = self.get_central_act_idx
        central_act_stroke = self.get_act_stroke_from_wf(
            self.wfs[central_act_idx])
        n_of_act = len(self.act_list)
        stroke_vector = np.zeros(n_of_act)
        for idx in range(n_of_act):
            yc, xc = self.get_act_stroke_coord_from_wf(self.wfs[idx])
            stroke_vector[idx] = self.wfs[central_act_idx, yc, xc]
        return stroke_vector / central_act_stroke

    @property
    def get_central_act_idx(self):
        return np.argwhere(self.central_act == self.act_list)[0][0]
