import numpy as np
from tesi_ao.main220316 import create_devices
from tesi_ao.mems_command_to_position_linearization_analyzer import CommandToPositionLinearizationAnalyzer


class FlatStabilityAnalyzer():
    ffmt = '.fits'
    fpath = 'prova/misure_con_tappo/misure_ripetute_bozzo/trm_'
    frame_shape = (486, 640)

    def __init__(self):
        self._wyko, self._bmc = create_devices()
        self.n_of_flats = 140

    def load_deformed_flats_from_file(self, act_list=None, Ntimes=3):

        if act_list is None:
            act_list = np.arange(self._bmc.get_number_of_actuators())
            n_acts = len(act_list)
        if type(act_list) == int:
            act_list = np.array([act_list])
            n_acts = 1
        if type(act_list) == np.ndarray:
            act_list = np.array(act_list)
            n_acts = len(act_list)

        self.wfs_flat = np.ma.zeros(
            (Ntimes, n_acts, self.frame_shape[0], self.frame_shape[1]))
        for times in np.arange(Ntimes):
            print('%dtimes:' % times)
            for act_idx, act in enumerate(act_list):
                print('Loading act#%d from file' % int(act))
                fname = self.fpath + \
                    'act%d' % int(act) + 'time%d' % times + self.ffmt
                cpla = CommandToPositionLinearizationAnalyzer(fname)
                self.wfs_flat[times, act_idx] = cpla._acquired_wfflat

    def load_flats_from_file(self):
        act_list = np.array([15, 101, 113])
        n_acts = len(act_list)
        Ntimes = 10

        self.wfs_flat = np.ma.zeros(
            (Ntimes, n_acts, self.frame_shape[0], self.frame_shape[1]))
        for times in np.arange(Ntimes):
            print('%dtimes:' % times)
            for act_idx, act in enumerate(act_list):
                print('Loading act#%d from file' % int(act))
                fpath = 'prova/misure_con_tappo/debozzati_15_113/trm_'
                fname = fpath + \
                    'act%d' % int(act) + 'time%d' % times + self.ffmt
                cpla = CommandToPositionLinearizationAnalyzer(fname)
                self.wfs_flat[times, act_idx] = cpla._acquired_wfflat

    def collapse_wf_map(self):
        total_of_flats = self.wfs_flat.shape[0] * self.wfs_flat.shape[1]
        wfs_all = self.wfs_flat.reshape(
            total_of_flats, self.frame_shape[0], self.frame_shape[1])
        mean_wf_map = wfs_all.mean(axis=0)
        sigma_wf_map = wfs_all.std(axis=0)
        amp_mean_wf_map = mean_wf_map.std()
        amp_i = wfs_all.std(axis=(1, 2))
        err_on_amp_mean_wf_map = amp_i.std()
        print(amp_mean_wf_map)
        print(err_on_amp_mean_wf_map)
        return mean_wf_map, sigma_wf_map
