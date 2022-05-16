
import numpy as np
from plico_interferometer import interferometer
from plico_dm import deformableMirror
from astropy.io import fits
from scipy.interpolate.interpolate import interp1d
from functools import reduce
from tesi_ao.mems_command_linearization import MemsCommandLinearization


def create_devices():
    wyko = interferometer('193.206.155.29', 7300)
    bmc = deformableMirror('193.206.155.92', 7000)
    return wyko, bmc


class CommandToPositionLinearizationMeasurerTRASHME(object):

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

    def _get_wavefront_flat_subtracted(self):
        dd = self._interf.wavefront(
            self.NUMBER_WAVEFRONTS_TO_AVERAGE) - self._get_zero_command_wavefront()
        return dd - np.ma.median(dd)

    def _reset_flat_wavefront(self):
        self._wfflat = None

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


class CommandToPositionLinearizationAnalyzerTRASHME(object):

    def __init__(self, scan_fname):
        res = CommandToPositionLinearizationMeasurer.load(scan_fname)
        self._wfs = res['wfs']
        self._cmd_vector = res['cmd_vector']
        self._actuators_list = res['actuators_list']
        self._reference_shape_tag = res['reference_shape_tag']
        self._n_steps_voltage_scan = self._wfs.shape[1]

    def _max_wavefront(self, act_idx, cmd_index):
        wf = self._wfs[act_idx, cmd_index]
        coord_max = np.argwhere(np.abs(wf) == np.max(np.abs(wf)))[0]
        return wf[coord_max[0], coord_max[1]]

    def _max_roi_wavefront(self, act_idx, cmd_index):
        wf = self._wfs[act_idx, cmd_index]
        b, t, l, r = self._get_max_roi(act_idx)
        wfroi = wf[b:t, l:r]
        coord_max = np.argwhere(
            np.abs(wfroi) == np.max(np.abs(wfroi)))[0]
        return wfroi[coord_max[0], coord_max[1]]

    def _get_max_roi(self, act):
        roi_size = 50
        wf = self._wfs[act, 0]
        coord_max = np.argwhere(np.abs(wf) == np.max(np.abs(wf)))[0]
        return coord_max[0] - roi_size, coord_max[0] + roi_size, \
            coord_max[1] - roi_size, coord_max[1] + roi_size

    def _max_vector(self, act_idx):
        res = np.zeros(self._n_steps_voltage_scan)
        for i in range(self._n_steps_voltage_scan):
            res[i] = self._max_roi_wavefront(act_idx, i)
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


class Robaccia220223(object):

    def __init__(self, cmd_vector, amplitude_vector):
        x = cmd_vector
        y = amplitude_vector
        res = np.polyfit(x, y, 2)
        self.a = res[0]
        self.b = res[1]
        self.c = res[2]

    def quadratic_fit(self):
        n_acts_to_meas = len(self._actuators_list)
        self._quadratic_coeffs = np.zeros((n_acts_to_meas, 3))

        for index in range(n_acts_to_meas):
            x = self._cmd_vector[index]
            y = self._max_vector(index)
            res = np.polyfit(x, y, 2)
            self._quadratic_coeffs[index] = res

    def _get_quadratic_coeffs(self, act):
        actidx = np.argwhere(self._actuators_list == act)[0][0]
        a = self._quadratic_coeffs[actidx, 0]
        b = self._quadratic_coeffs[actidx, 1]
        c = self._quadratic_coeffs[actidx, 2]
        return a, b, c

    def c2p(self, act, v):
        a, b, c = self._get_quadratic_coeffs(act)
        return a * v**2 + b * v + c

    def p2c(self, act, p):
        a, b, c = self._get_quadratic_coeffs(act)
        v = (-b - np.sqrt(b**2 - 4 * a * (c - p))) / (2 * a)
        return v


# provata con il file cpl_all fatto il 28/2 che ti ho messo sulla chiavetta
def provarec(cplm, cpla):
    # maschera intersezione di tutte le maschere delle wf map misurate
    imask = reduce(lambda a, b: np.ma.mask_or(a, b), cplm._wfs[:, 7].mask)

    # normalizzo la mappa di fase dell'attuatore act a max==1
    # scelgo il comando 7: è una deformata di circa 500nm (quindi ben sopra al
    # rumore dell'interferometro)
    # e non troppo grossa da saturare l'interferometro: per tutti i 140
    # attuatori la mappa del comando 7 non presenta "buchi"
    #
    # la funzione ritorna un vettore contenente i valori del wf nei punti
    # dentro la maschera imask
    def normalizeif(act):
        return (cplm._wfs[act, 7][imask == False] / cpla._max_deflection[act, 7]).data

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
    wffitted = np.zeros((486, 640))
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
