import numpy as np


class MemsZonalReconstructor(object):

    def __init__(self, cmask, ifs_stroke, act_list, ifs):
        self._cmask = cmask
        self._frame_shape = cmask.shape
        self._ifs = ifs
        self._check_input_mask()

        self._ifs[:].mask = cmask
        self._ifs_stroke = ifs_stroke
        self._act_list = act_list
        self._num_of_acts = len(act_list)

        self._normalize_influence_function()
        self._build_interaction_matrix()
        self.u, self.s, self.vh = np.linalg.svd(
            self._im, full_matrices=False)
        self._build_reconstruction_matrix_via_pinv()

    def _check_input_mask(self):
        '''
        Checks cmask dimensions
        True if cmask is fully inscribed in the ifs mask
        '''
        assert self._cmask.shape == self._ifs[
            0].shape, "cmask has not the same dimension of ifs mask!\nGot:{self._cmask.shape}\nShould be:{self._ifs[0].shape}"

        ifs_mask = self._ifs[0].mask
        intersection_mask = np.ma.mask_or(ifs_mask, self._cmask)
        assert (intersection_mask == self._cmask).all(
        ) == True, "input mask is not valid!\nShould be fully inscribed in the ifs mask!"

    def _normalize_influence_function(self):
        # normalizzare al pushpull o al max registarto nel pixel
        self._normalized_ifs = self._ifs[:] / self._ifs_stroke

    def _build_interaction_matrix(self):
        self._im = np.column_stack([self._normalized_ifs[act][self._cmask == False]
                                    for act in self._act_list])

    def _build_reconstruction_matrix_via_pinv(self):
        self._rec = np.linalg.pinv(self._im)

    def get_reconstruction_matrix(self):
        return self._rec

    def _build_reconstruction_matrix_via_svd(self, eigenvalue_to_use):

        large = np.zeros(self.s.shape).astype(bool)
        large[eigenvalue_to_use] = True
        s = np.divide(1, self.s, where=large)
        s[~large] = 0
        self._rec_svd = np.matmul(np.transpose(self.vh), np.multiply(
            s[..., np.newaxis], np.transpose(self.u)))
