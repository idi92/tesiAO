import numpy as np

class Boston140Display():
    
    def __init__(self):
        self._n_acts = 140
        self._size = 12
        a=np.arange(self._size**2); 
        self._indexer = np.delete(a, [0,11,132, 143])
    

    def map(self, cmd_or_pos):
        mm = np.zeros(self._size**2)
        mm[self._indexer] = cmd_or_pos
        return np.flipud(mm.reshape((self._size, self._size)).T) 