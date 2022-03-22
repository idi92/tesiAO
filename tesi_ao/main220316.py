
import numpy as np
from plico_interferometer import interferometer
from plico_dm import deformableMirror
from astropy.io import fits
from scipy.interpolate.interpolate import interp1d
from functools import reduce
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def create_devices():
    wyko = interferometer('193.206.155.29', 7300)
    bmc = deformableMirror('193.206.155.92', 7000)
    return wyko, bmc


class CommandToPositionLinearizationMeasurer(object):

    NUMBER_WAVEFRONTS_TO_AVERAGE = 1
    NUMBER_STEPS_VOLTAGE_SCAN = 11

    def __init__(self, interferometer, mems_deformable_mirror):
        self._interf = interferometer
        self._bmc = mems_deformable_mirror
        self._n_acts = self._bmc.get_number_of_actuators()
        self._wfflat = None

    def _get_zero_command_wavefront(self):
        if self._wfflat is None:
            cmd = np.zeros(self._n_acts)
            self._bmc.set_shape(cmd)
            self._wfflat = self._interf.wavefront(
                self.NUMBER_WAVEFRONTS_TO_AVERAGE)
        return self._wfflat

    def execute_command_scan(self, act_list=None):
        if act_list is None:
            act_list = np.arange(self._n_acts)

        self._actuators_list = np.array(act_list)
        n_acts_to_meas = len(self._actuators_list)

        wfflat = self._get_zero_command_wavefront()

        self._reference_cmds = self._bmc.get_reference_shape()
        self._reference_tag = self._bmc.get_reference_shape_tag()

        self._cmd_vector = np.zeros((n_acts_to_meas,
                                     self.NUMBER_STEPS_VOLTAGE_SCAN))

        self._wfs = np.ma.zeros(
            (n_acts_to_meas, self.NUMBER_STEPS_VOLTAGE_SCAN,
             wfflat.shape[0], wfflat.shape[1]))

        N_pixels = self._wfs.shape[2] * self._wfs.shape[3]
        for act_idx, act in enumerate(self._actuators_list):
            self._cmd_vector[act_idx] = np.linspace(
                0, 1, self.NUMBER_STEPS_VOLTAGE_SCAN) - self._reference_cmds[act]
            for cmd_idx, cmdi in enumerate(self._cmd_vector[act_idx]):
                print("Act:%d - command %g" % (act, cmdi))
                cmd = np.zeros(self._n_acts)
                cmd[act] = cmdi
                self._bmc.set_shape(cmd)
                self._wfs[act_idx, cmd_idx, :,
                          :] = self._get_wavefront_flat_subtracted()
                masked_pixels = self._wfs[act_idx, cmd_idx].mask.sum()
                masked_ratio = masked_pixels / N_pixels
                if masked_ratio > 0.7829:
                    print('Warning: Bad measure acquired for: act%d' %
                          act_idx + ' cmd_idx %d' % cmd_idx)
                    self._avoid_saturated_measures(
                        masked_ratio, act_idx, cmd_idx, N_pixels)

    def _avoid_saturated_measures(self, masked_ratio, act_idx, cmd_idx, N_pixels):

        while masked_ratio > 0.7829:
            self._wfs[act_idx, cmd_idx, :,
                      :] = self._get_wavefront_flat_subtracted()
            masked_pixels = self._wfs[act_idx, cmd_idx].mask.sum()
            masked_ratio = masked_pixels / N_pixels

        print('Repeated measure completed!')

    def _get_wavefront_flat_subtracted(self):
        dd = self._interf.wavefront(
            self.NUMBER_WAVEFRONTS_TO_AVERAGE) - self._get_zero_command_wavefront()
        return dd - np.ma.median(dd)

    def _reset_flat_wavefront(self):
        self._wfflat = None

    def check_mask_coverage(self, ratio=False):
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

    def add_repeated_measure(self, cplm_to_add, act_list):
        for idx, act in enumerate(act_list):
            self._wfs[act] = cplm_to_add._wfs[idx]

    def save_results(self, fname):
        hdr = fits.Header()
        hdr['REF_TAG'] = self._reference_tag
        hdr['N_AV_FR'] = self.NUMBER_WAVEFRONTS_TO_AVERAGE
        fits.writeto(fname, self._wfs.data, hdr)
        fits.append(fname, self._wfs.mask.astype(int))
        fits.append(fname, self._cmd_vector)
        fits.append(fname, self._actuators_list)
        fits.append(fname, self._reference_cmds)

    @staticmethod
    def load(fname):
        header = fits.getheader(fname)
        hduList = fits.open(fname)
        wfs_data = hduList[0].data
        wfs_mask = hduList[1].data.astype(bool)
        wfs = np.ma.masked_array(data=wfs_data, mask=wfs_mask)
        cmd_vector = hduList[2].data
        actuators_list = hduList[3].data
        reference_commands = hduList[4].data
        return {'wfs': wfs,
                'cmd_vector': cmd_vector,
                'actuators_list': actuators_list,
                'reference_shape': reference_commands,
                'reference_shape_tag': header['REF_TAG']
                }


class CommandToPositionLinearizationAnalyzer(object):

    def __init__(self, scan_fname):
        res = CommandToPositionLinearizationMeasurer.load(scan_fname)
        self._wfs = res['wfs']
        self._cmd_vector = res['cmd_vector']
        self._actuators_list = res['actuators_list']
        self._reference_shape_tag = res['reference_shape_tag']
        self._n_steps_voltage_scan = self._wfs.shape[1]

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
    # def _get_max_roi(self, act):
    #     roi_size = 50
    #     wf = self._wfs[act, 0]
    #     coord_max = np.argwhere(np.abs(wf) == np.max(np.abs(wf)))[0]
    #     return coord_max[0] - roi_size, coord_max[0] + roi_size, \
    #         coord_max[1] - roi_size, coord_max[1] + roi_size

    def _get_max_pixel(self, act):
        wf = self._wfs[act, 2]
        coord_max = np.argwhere(np.abs(wf) == np.max(np.abs(wf)))[0]
        return coord_max[0], coord_max[1]

    def _max_wavefront(self, act, cmd_index):
        wf = self._wfs[act, cmd_index]
        y, x = self._get_max_pixel(act)
        return wf[y, x]

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

    def add_repeated_measure(self, cpla_to_add, act_list):
        for idx, act in enumerate(act_list):
            self._wfs[act] = cpla_to_add._wfs[idx]

    def check_mask_coverage(self, ratio=False):
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

    # def _2dgaussian(self, X, amplitude, x0, y0, sigmax, sigmay, offset):
    #     y, x = X
    #     z = np.zeros((len(y), len(x)), dtype='float')
    #     N = amplitude  # *0.5 / (np.pi * sigmax * sigmay)
    #     for xi in np.arange(len(x)):
    #         a = 0.5 * ((xi - x0) / sigmax)**2
    #         for yi in np.arange(len(y)):
    #             b = 0.5 * ((yi - y0) / sigmay)**2
    #
    #             z[yi, xi] = N * np.exp(-(a + b)) + offset
    #     return z.ravel()
    #
    # def _gaussian_fitting(self, act_idx, cmd_index):
    #     wf = self._wfs[act_idx, cmd_index]
    #     b, t, l, r = self._get_max_roi(act_idx)
    #     wfroi = wf[b:t, l:r]
    #     z = wfroi
    #     x = np.arange(wfroi.shape[1], dtype='float')
    #     y = np.arange(wfroi.shape[0], dtype='float')
    #
    #     A0 = self._max_roi_wavefront(act_idx, cmd_index)
    #     coord_max = np.argwhere(np.abs(wfroi) == np.max(np.abs(wfroi)))[0]
    #     x0 = coord_max[1]
    #     y0 = coord_max[0]
    #     sigma0 = 25.
    #     sigmax = sigma0
    #     sigmay = sigma0
    #     offset = 0.
    #     starting_values = [A0, x0, y0, sigmax, sigmay, offset]
    #     X = y, x
    #
    #     Z = np.zeros((len(y), len(x)), dtype='float')
    #     for j in np.arange(len(y)):
    #         prova = z[j].compressed()
    #         Z[j] = prova
    #
    #     #err_z = Z.std() * np.ones(len(x) * len(y))
    #
    #     fpar, fcov = curve_fit(self._2dgaussian, X,
    #                            Z.ravel(), p0=starting_values, absolute_sigma=True)
    #     #err_fpar = np.sqrt(np.diag(fcov))
    #     error = (Z.ravel() - self._2dgaussian(X, *fpar))
    #     starting_values = [fpar[0], fpar[1],
    #                        fpar[2], fpar[3], fpar[4], fpar[5]]
    #     fpar, fcov = curve_fit(self._2dgaussian, X,
    #                            Z.ravel(), p0=starting_values, sigma=error, absolute_sigma=True)
    #
    #     return fpar[0]
    #
    # def _compute_gaussian_amplitude_deflection(self):
    #     self._max_deflection = np.zeros(
    #         (self._cmd_vector.shape[0], self._cmd_vector.shape[1]))
    #     for act in range(self._cmd_vector.shape[0]):
    #         for cmd_idx in range(self._cmd_vector.shape[1]):
    #             self._max_deflection[act, cmd_idx] = self._gaussian_fitting(
    #                 act, cmd_idx)
    #
    # def compute_gaussian_linearization(self):
    #     self._compute_gaussian_amplitude_deflection()
    #
    #     return MemsCommandLinearization(
    #         self._actuators_list,
    #         self._cmd_vector,
    #         self._max_deflection,
    #         self._reference_shape_tag)


class MemsCommandLinearization():

    def __init__(self,
                 actuators_list,
                 cmd_vector,
                 deflection,
                 reference_shape_tag):
        self._actuators_list = actuators_list
        self._cmd_vector = cmd_vector
        self._deflection = deflection
        self._reference_shape_tag = reference_shape_tag
        self._create_interpolation()

    def _create_interpolation(self):
        self._finter = [interp1d(
            self._deflection[i], self._cmd_vector[i], kind='cubic')
            for i in range(self._cmd_vector.shape[0])]

    def _get_act_idx(self, act):
        return np.argwhere(self._actuators_list == act)[0][0]

    def p2c(self, act, p):
        idx = self._get_act_idx(act)
        return self._finter[idx](p)

    def save(self, fname):
        hdr = fits.Header()
        hdr['REF_TAG'] = self._reference_shape_tag
        fits.writeto(fname, self._actuators_list, hdr)
        fits.append(fname, self._cmd_vector)
        fits.append(fname, self._deflection)

    @staticmethod
    def load(fname):
        header = fits.getheader(fname)
        hduList = fits.open(fname)
        actuators_list = hduList[0].data
        cmd_vector = hduList[1].data
        deflection = hduList[2].data
        reference_shape_tag = header['REF_TAG']
        return MemsCommandLinearization(
            actuators_list, cmd_vector, deflection, reference_shape_tag)


def main220228():
    mcl = MemsCommandLinearization.load('/tmp/mcl9.fits')
    print('reference shape used when calibrating %s ' %
          mcl._reference_shape_tag)
    actuator_number = 63
    deflection_wrt_reference_shape = 100e-9
    mcl.p2c(actuator_number, deflection_wrt_reference_shape)


def main_calibration(wyko,
                     bmc,
                     mcl_fname='/tmp/mcl0.fits',
                     scan_fname='/tmp/cpl0.fits',
                     act_list=None):
    #wyko, bmc = create_devices()
    cplm = CommandToPositionLinearizationMeasurer(wyko, bmc)

    if act_list is None:
        act_list = np.arange(bmc.get_number_of_actuators())
    cplm.execute_command_scan(act_list)
    cplm.save_results(scan_fname)
    cpla = CommandToPositionLinearizationAnalyzer(scan_fname)
    mcl = cpla.compute_linearization()
    mcl.save(mcl_fname)
    return mcl, cplm, cpla


def plot_interpolated_function(mcl):
    plt.figure()
    plt.clf()
    for idx, act in enumerate(mcl._actuators_list):
        a = np.min(mcl._deflection[act])
        b = np.max(mcl._deflection[act])
        xx = np.linspace(a, b, 1000)
        plt.plot(mcl._finter[act](xx), xx / 1.e-9, '.-')
    plt.xlabel('Command [au]', size=25)
    plt.ylabel('Deflection [nm]', size=25)
    plt.title('Calibration curve per actuator', size=25)
    plt.grid()


def plot_acquired_measures(mcl):
    plt.figure()
    plt.clf()
    for idx, act in enumerate(mcl._actuators_list):
        plt.plot(mcl._cmd_vector[idx], mcl._deflection[idx] / 1.e-9, '.-')
    plt.xlabel('Command [au]', size=25)
    plt.ylabel('Deflection [nm]', size=25)
    plt.title('Acquired Measures per actuator', size=25)
    plt.grid()


def plot_single_curve(mcl, act):
    plt.figure()
    plt.clf()
    a = np.min(mcl._deflection[act])
    b = np.max(mcl._deflection[act])
    xx = np.linspace(a, b, 1000)
    plt.plot(mcl._cmd_vector[act], mcl._deflection[act] /
             1.e-9, 'or', label='sampling points')
    plt.plot(mcl._finter[act](xx), xx / 1.e-9, '-', label='finter')
    plt.title('Calibration Curve: act#%d' % act, size=25)
    plt.xlabel('Commands [au]', size=25)
    plt.ylabel('Deflection [nm]', size=25)
    plt.grid()
    plt.legend(loc='best')


# da provare sul file cplm_all_fixed fatto il 17/3

class ModeGenerator():

    NORM_AT_THIS_CMD = 13  # such that wyko noise and saturation are avoided

    def __init__(self, cpla, mcl):
        self._cpla = cpla
        self._mcl = mcl

    def build_intersection_mask(self):
        self._imask = reduce(lambda a, b: np.ma.mask_or(
            a, b), self._cpla._wfs[:, self.NORM_AT_THIS_CMD].mask)

    def _normalize_influence_function(self, act):
        return (self._cpla._wfs[act, self.NORM_AT_THIS_CMD][self._imask == False] / self._mcl._deflection[act, self.NORM_AT_THIS_CMD]).data

    def build_interaction_matrix(self):
        self._im = np.column_stack([self._normalize_influence_function(
            act) for act in self._cpla._actuators_list])

    def build_reconstruction_matrix(self):
        self._rec = np.linalg.pinv(self._im)

    def generate_mode(self, mode):
        self._wfmode = np.ma.array(data=mode, mask=self._imask)

    def generate_tilt(self):
        self._wfmode = np.tile(np.linspace(-100e-9, 100e-9, 640), (486, 1))
        self._wfmode = np.ma.array(data=self._wfmode, mask=self._imask)

    def get_position_cmds_from_wf(self, wfmap=None):
        if wfmap is None:
            wfmap = self._wfmode
        pos = np.dot(self._rec, wfmap[self._imask == False])
        # check and clip cmds
        for act, stroke in enumerate(pos):
            max_stroke = np.max(self._mcl._deflection[act])
            min_stroke = np.min(self._mcl._deflection[act])
            if(stroke > max_stroke):
                pos[act] = max_stroke
                print('act%d reached max stroke' % act)
            if(stroke < min_stroke):
                pos[act] = min_stroke
                print('act%d reached min stroke' % act)
        return pos

    def build_fitted_wavefront(self, wfmap=None):
        if wfmap is None:
            wfmap = self._wfmode
        pos_from_wf = self.get_position_cmds_from_wf(wfmap)
        self._wffitted = np.zeros(
            (self._cpla._wfs.shape[2], self._cpla._wfs.shape[3]))
        self._wffitted[self._imask == False] = np.dot(self._im, pos_from_wf)
        self._wffitted = np.ma.array(data=self._wffitted, mask=self._imask)

    def plot_generated_and_expected_WF(self):
        plt.figure()
        plt.clf()
        plt.imshow(self._wfmode)
        plt.colorbar()
        plt.title('Generated Mode', size=25)
        plt.figure()
        plt.clf()
        plt.imshow(self._wffitted)
        plt.colorbar()
        plt.title('Fitted Mode', size=25)
        plt.figure()
        plt.clf()
        plt.imshow(self._wffitted - self._wfmode)
        plt.colorbar()
        plt.title('Mode difference', size=25)

        print("Expectations:")
        print("mode amplitude: %g m rms " % self._wfmode.std())
        fitting_error = (self._wffitted - self._wfmode).std()
        print("fitting error: %g m rms " % fitting_error)


class ModeMeasurer():

    def __init__(self, interferometer, mems_deformable_mirror):
        self._interf = interferometer
        self._bmc = mems_deformable_mirror

    def execute_measure(self, pos, mcl):
        flat_cmd = np.zeros(self._bmc.get_number_of_actuators())
        self._bmc.set_shape(flat_cmd)
        wfflat = self._interf.wavefront()

        cmd = np.zeros(self._bmc.get_number_of_actuators())
        for act, stroke in enumerate(pos.data):
            cmd[act] = mcl.p2c(act, stroke)

        self._bmc.set_shape(cmd)
        #_get_wavefront_flat_subtracted
        wfflatsub = self._interf.wavefront() - wfflat
        self._wfmeas = wfflatsub - np.ma.median(wfflatsub)

    def plot_expected_and_measured_mode(self, wffitted):
        plt.figure()
        plt.clf()
        plt.ion()
        plt.imshow(self._wfmeas)
        plt.colorbar()
        plt.title('Observed Mode', size=25)
        plt.figure()
        plt.clf()
        plt.ion()
        plt.imshow(self._wfmeas - wffitted)
        plt.colorbar()
        plt.title('Difference Observed-Expected', size=25)
        print("Observation:")
        amp_mode = self._wfmeas.std()
        amp_mode = amp_mode / 1.e-9
        print("mode amplitude: %g nm rms " % amp_mode)
        fitting_meas_error = (self._wfmeas - wffitted).std()
        fitting_meas_error = fitting_meas_error / 1.e-9
        print("fitting error: %g nm rms " % fitting_meas_error)


def provarec(cpla, mcl):
    # maschera intersezione di tutte le maschere delle wf map misurate
    imask = reduce(lambda a, b: np.ma.mask_or(a, b), cpla._wfs[:, 13].mask)

    # normalizzo la mappa di fase dell'attuatore act a max==1
    # scelgo il comando 13: è una deformata di circa 500nm (quindi ben sopra al
    # rumore dell'interferometro)
    # e non troppo grossa da saturare l'interferometro: per tutti i 140
    # attuatori la mappa del comando 13 non presenta "buchi"
    #
    # la funzione ritorna un vettore contenente i valori del wf nei punti
    # dentro la maschera imask
    def normalizeif(act):
        return (cpla._wfs[act, 13][imask == False] / mcl._deflection[act, 13]).data

    # creo una "matrice di interazione" giustapponendo in colonna i vettori
    # normalizzati
    im = np.column_stack([normalizeif(i) for i in cpla._actuators_list])

    # pseudo inversa della matrice di interazione
    rec = np.linalg.pinv(im)

    # questo prodotto matriciale fra rec e una mappa di fase qualsiasi restituisce
    # le 140 posizioni in cui comandare gli attuatori per ottenere wfmap
    def wf2pos(wfmap):
        return np.dot(rec, wfmap[imask == False])

    # creo un tilt (da -100 a 100nm lungo ogni riga, tutte le righe sono uguali
    wftilt = np.tile(np.linspace(-100e-9, 100e-9, 640), (486, 1))
    # lo converto in masked_array per coerenza col resto
    wftilt = np.ma.array(data=wftilt, mask=imask)

    # postilt è il comando da dare agli attuatori per ottenere wftilt
    postilt = wf2pos(wftilt)
    # bisognerebbe controllare che nessun elemento sia troppo grande,
    # altrimenti lo specchio non riesce a fare la deformata richiesta

    # wffitted è la mappa di fase che mi aspetto di ottenere davvero:
    # è il miglior fit che lo specchio riesce a fare di wftilt
    wffitted = np.zeros((cpla._wfs.shape[2], cpla._wfs.shape[3]))
    wffitted[imask == False] = np.dot(im, postilt)
    wffitted = np.ma.array(data=wffitted, mask=imask)

    print("mode amplitude: %g nm rms " % wftilt.std())
    fitting_error = (wffitted - wftilt).std()
    print("fitting error: %g nm rms " % fitting_error)

    # posso rifarlo con un modo ad alta frequenza (tipo un seno che oscilla 15
    # volte in 640 px)
    wfsin = np.tile(100e-9 * np.sin(2 * np.pi / 43 * np.arange(640)), (486, 1))
    wfsin = np.ma.array(data=wfsin, mask=imask)

    possin = wf2pos(wfsin)

    sinfitted = np.zeros((486, 640))
    sinfitted[imask == False] = np.dot(im, possin)
    sinfitted = np.ma.array(data=sinfitted, mask=imask)

    # il fitting error su questo modo è 2nm rms
    fitting_error_sin = (sinfitted - wfsin).std()

    print("mode amplitude sin: %g nm rms " % wfsin.std())
    print("fitting error: %g nm rms " % fitting_error_sin)
