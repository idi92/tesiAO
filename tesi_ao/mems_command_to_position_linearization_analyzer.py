import numpy as np
from tesi_ao.mems_command_linearization import MemsCommandLinearization
from tesi_ao.mems_command_to_position_linearization_measurer import \
    CommandToPositionLinearizationMeasurer


class CommandToPositionLinearizationAnalyzer(object):

    def __init__(self, scan_fname):
        res = CommandToPositionLinearizationMeasurer.load(scan_fname)
        self._wfs = res['wfs']
        self._cmd_vector = res['cmd_vector']
        self._actuators_list = res['actuators_list']
        self._reference_shape_tag = res['reference_shape_tag']
        self._n_steps_voltage_scan = self._wfs.shape[1]
        # TODO: aggiungere try per caricare le misure del wf_flat
        # dal file nel caso le possieda o meno
        try:
            self._acquired_wfflat = res['wfs_flat']
        except KeyError:
            pass
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
