import numpy as np
from tesi_ao.main220316 import create_devices
from tesi_ao.mems_command_to_position_linearization_analyzer import CommandToPositionLinearizationAnalyzer
from astropy.io import fits


class FlatStabilityAnalyzer():
    ffmt = '.fits'
    fpath = 'prova/misure_con_tappo/misure_ripetute_bozzo/trm_'
    frame_shape = (486, 640)

    def __init__(self):
        self._wyko, self._bmc = create_devices()
        self.n_of_flats = 140

    def load_deformed_flats_from_file(self, act_list=None, Ntimes=3):
        '''
        file on
        'prova/misure_con_tappo/misure_ripetute_bozzo'
        deformed flat

        '''

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

    def load_flat_220616(self):
        '''
        file on
        'prova/mems_interpolation_error'
        normal flat
        '''
        act_list = np.array([27, 60, 63, 67, 76, 111])
        scan_list = np.array([10, 20, 30, 40, 50, 60, 100])
        n_act = len(act_list)
        n_scan = len(scan_list)
        self.n_flats = n_act * n_scan
        frame_shape = (486, 640)
        self.wfs_flat = np.ma.zeros(
            (self.n_flats, frame_shape[0], frame_shape[1]))

        idx = 0
        for act in act_list:
            print('Loading act#%d from file' % int(act))
            for scan in scan_list:
                fname = 'prova/mems_interpolation_error/iea_cplm_act%d' % act + \
                    '_nscans%d' % scan + '_220616.fits'
                cpla = CommandToPositionLinearizationAnalyzer(fname)
                self.wfs_flat[idx] = cpla._acquired_wfflat
                idx += 1

    def collapse_normal_flat_maps(self):
        mean_wf_map = self.wfs_flat.mean(axis=0)
        sigma_wf_map = self.wfs_flat.std(axis=0)
        mean_amp = self.wfs_flat.std(axis=(1, 2)).mean()
        err_amp = self.wfs_flat.std(axis=(1, 2)).std()
        print(mean_amp)
        print(err_amp)
        return mean_wf_map, sigma_wf_map

    def save_normal_flat_map(self, fname):
        num_of_wfs = self.n_flats
        mean_wf_map, sigma_wf_map = self.collapse_normal_flat_maps()
        hdr = fits.Header()
        hdr['NWFS'] = num_of_wfs
        fits.writeto(fname, mean_wf_map.data, hdr)
        fits.append(fname, mean_wf_map.mask.astype(int))
        fits.append(fname, sigma_wf_map.data)
        fits.append(fname, sigma_wf_map.mask.astype(int))

    def collapse_deformed_flat_maps(self):
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

    def save_deformed_flat_map(self, fname):
        num_of_wfs = self.wfs_flat.shape[0] * self.wfs_flat.shape[1]
        mean_wf_map, sigma_wf_map = self.collapse_deformed_flat_maps()
        hdr = fits.Header()
        hdr['NWFS'] = num_of_wfs
        fits.writeto(fname, mean_wf_map.data, hdr)
        fits.append(fname, mean_wf_map.mask.astype(int))
        fits.append(fname, sigma_wf_map.data)
        fits.append(fname, sigma_wf_map.mask.astype(int))

    @staticmethod
    def load_map(fname):
        header = fits.getheader(fname)
        hduList = fits.open(fname)
        num_of_wfs = header['NWFS']
        wfm_data = hduList[0].data
        wfm_mask = hduList[1].data.astype(bool)
        wf_mean = np.ma.array(data=wfm_data, mask=wfm_mask)
        wfs_data = hduList[2].data
        wfs_mask = hduList[3].data.astype(bool)
        wf_sigma = np.ma.array(data=wfs_data, mask=wfs_mask)
        return wf_mean, wf_sigma, num_of_wfs
