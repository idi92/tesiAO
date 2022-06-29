import numpy as np


class MemsZonalReconstructor(object):

    THRESHOLD_RMS = 0.15  # threshold for wf rms to select actuators outside the specified mask

    def __init__(self, cmask, ifs_stroke, ifs):
        self._cmask = cmask
        self._ifs = ifs
        self._check_input_mask()

        self._ifs[:].mask = cmask
        self._ifs_stroke = ifs_stroke
        self._num_of_acts = ifs.shape[0]

        self._normalize_influence_function()
        self._reset()

    def _reset(self):
        self._acts_in_pupil = None
        self._im = None
        self._rec = None

    def _compute_all(self):
        self._build_valid_actuators_list()
        self._build_interaction_matrix()
        self._build_reconstruction_matrix_via_pinv()

    def _get_svd(self):
        self.u, self.s, self.vh = np.linalg.svd(
            self.interaction_matrix, full_matrices=False)

    def _check_input_mask(self):
        '''
        Checks cmask dimensions
        True if cmask is fully inscribed in the ifs mask
        '''
        assert self.mask.shape == self._ifs[
            0].shape, "cmask has not the same dimension of ifs mask!\nGot:{self.mask.shape}\nShould be:{self._ifs[0].shape}"

        ifs_mask = self._ifs[0].mask
        intersection_mask = np.ma.mask_or(ifs_mask, self.mask)
        assert (intersection_mask == self.mask).all(
        ) == True, "input mask is not valid!\nShould be fully inscribed in the ifs mask!"

    def _normalize_influence_function(self):
        # normalizzare al pushpull o al max registarto nel pixel
        self._normalized_ifs = self._ifs[:] / self._ifs_stroke

    def _build_interaction_matrix(self):
        self._im = np.column_stack([self._normalized_ifs[act][self.mask == False]
                                    for act in self.selected_actuators])

    @property
    def interaction_matrix(self):
        if self._im is None:
            self._build_interaction_matrix()
        return self._im

    def _build_reconstruction_matrix_via_pinv(self):
        self._rec = np.linalg.pinv(self.interaction_matrix)

    @property
    def reconstruction_matrix(self):
        if self._rec is None:
            self._build_reconstruction_matrix_via_pinv()
        return self._rec

    def _build_reconstruction_matrix_via_svd(self, eigenvalue_to_use):

        large = np.zeros(self.s.shape).astype(bool)
        large[eigenvalue_to_use] = True
        s = np.divide(1, self.s, where=large)
        s[~large] = 0
        self._rec_svd = np.matmul(np.transpose(self.vh), np.multiply(
            s[..., np.newaxis], np.transpose(self.u)))

    def _build_valid_actuators_list(self):
        self._check_actuators_visibility_with_wf_rms()
        self._acts_in_pupil = np.where(
            self._rms_wf > self.THRESHOLD_RMS * self._rms_wf.max())[0]

    def _check_actuators_visibility_with_wf_rms(self):
        self._rms_wf = np.zeros(self._num_of_acts)
        for act in range(self._num_of_acts):
            self._rms_wf[act] = np.ma.array(data=self._ifs[act].data,
                                            mask=self.mask).std()

    def _show_actuators_visibility(self):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.clf()
        plt.ion()
        plt.plot(self._rms_wf / 1.e-9, 'o', label='push/pull %g m' %
                 self._ifs_stroke)
        plt.xlabel('# actuator', size=10)
        plt.ylabel('Surface rms [nm]', size=10)
        plt.grid()
        plt.legend(loc='best')

        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot(self._rms_wf / 1.e-9, '.', label='push/pull %g m' %
                 self._ifs_stroke)
        ax2.plot(self._rms_wf / self._rms_wf.max(), '.')

        ax1.set_xlabel('# actuators', size=10)
        ax1.set_ylabel('Surface rms [nm]', size=10)
        ax2.set_ylabel('Normalized surface rms', size=10)

    @property
    def selected_actuators(self):
        if self._acts_in_pupil is None:
            self._build_valid_actuators_list()
        return self._acts_in_pupil

    @property
    def number_of_selected_actuators(self):
        return len(self._acts_in_pupil)

    @property
    def total_number_of_actuators(self):
        return self._num_of_acts

    @property
    def mask(self):
        return self._cmask
