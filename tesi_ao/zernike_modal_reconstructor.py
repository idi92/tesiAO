import numpy as np
from arte.utils.zernike_generator import ZernikeGenerator



class ZernikeToZonal():
    
    def __init__(self, zonal_reconstructor, circular_mask):
        self._mzr = zonal_reconstructor
        self._circular_mask = circular_mask
        self._n_zern_modes = 50
        self._compute_zernike_to_zonal_conversion()
        
    @property
    def circular_mask(self):
        return self._circular_mask
    
    @property
    def modes_list(self):
        return np.arange(2, self._n_zern_modes+1)
    
    def _compute_zernike_to_zonal_conversion(self):
        zg = ZernikeGenerator(self.circular_mask)
        zdict = zg.getZernikeDict(self.modes_list)
        
        self._zern2pos = np.zeros((self._mzr.number_of_selected_actuators,
                             self._n_zern_modes))
        for i, v in enumerate(zdict.values()):
            self._zern2pos[:,i] = np.dot(self._mzr.reconstruction_matrix,
                                       v.compressed())
            
            
    def convert_modal_to_zonal(self, zernike_modal_amplitudes):
        pos = np.zeros(self._mzr.total_number_of_actuators)
        pos[self._mzr.selected_actuators] = np.dot(self._zern2pos,
                                                   zernike_modal_amplitudes)
        return pos


