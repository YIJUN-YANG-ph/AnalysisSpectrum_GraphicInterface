import contextlib
import io
import os
import unittest
import warnings
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication

import components_AnalysisSpectrum as analysis


class ResonanceFittingAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def tearDown(self):
        plt.close('all')

    @staticmethod
    def fake_fit(fail_call):
        calls = 0

        def fit(nu_res, transmission_res, bounds):
            nonlocal calls
            current = calls
            calls += 1
            if current == fail_call:
                raise RuntimeError('simulated fit failure')
            frequency = np.mean(nu_res)
            return (np.array([0.1, 1e9, frequency]), None,
                    lambda nu, amplitude, width, center: np.ones_like(nu))

        return fit

    def test_successful_indices_remain_aligned_after_fit_failures(self):
        wavelength = pd.Series(np.linspace(1500, 1560, 301))
        transmission = pd.Series(np.full(len(wavelength), 0.9))
        indices = np.arange(30, 271, 30)
        properties = {'prominences': np.ones(len(indices)),
                      'FWHM': np.full(len(indices), 1e9)}
        for failed in (None, 0, 4, len(indices) - 1):
            with self.subTest(failed=failed):
                with patch('F_LorentzianModel.f_locatized', side_effect=self.fake_fit(failed)):
                    with contextlib.redirect_stdout(io.StringIO()):
                        params, _ = analysis.resonances_fitting(
                            wavelength, transmission, indices, properties)
                expected = np.delete(indices, failed) if failed is not None else indices
                np.testing.assert_array_equal(params['idx_peak'].to_numpy(), expected)
                self.assertEqual(len(params['Q_f']), len(expected))

    def test_analysis_q_plot_uses_only_successful_peak_positions(self):
        wavelength = pd.Series(np.linspace(1500, 1560, 2201))
        transmission = pd.Series(np.full(len(wavelength), 0.9))
        indices = np.arange(50, 2151, 50)
        properties = {'prominences': np.ones(len(indices)),
                      'FWHM': np.full(len(indices), 1e9)}
        data = pd.DataFrame({'wavelength_nm': wavelength, 'T_linear': transmission,
                             'nu_Hz': analysis.wl2nu(wavelength)})

        with patch.object(analysis, 'find_peaks_for_FSR',
                          return_value=(pd.DataFrame({'idx_peaks': indices}), properties)):
            with patch.object(analysis, 'RemoveOffset_Savgol',
                              return_value=(transmission, np.ones(len(wavelength)))):
                with patch('F_LorentzianModel.f_locatized', side_effect=self.fake_fit(20)):
                    with patch('F_FSR_Dispersion.F_FSR_Dispersion',
                               side_effect=RuntimeError('stopped after Q plot')):
                        with contextlib.redirect_stdout(io.StringIO()):
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter('always')
                                with self.assertRaisesRegex(RuntimeError, 'stopped after Q plot'):
                                    analysis.analysis_main(
                                        data, {'diameter': 200}, {}, {}, {}, {},
                                        {}, {'pump wl': 1550})

        q_axes = [axis for number in plt.get_fignums()
                  for axis in plt.figure(number).axes if axis.get_ylabel() == 'Q factor']
        self.assertEqual(len(q_axes), 1)
        line = q_axes[0].lines[0]
        np.testing.assert_array_equal(line.get_xdata(), wavelength.iloc[np.delete(indices, 20)])
        self.assertEqual(len(line.get_ydata()), 42)
        self.assertTrue(any('frequency differences can span missing peaks' in str(item.message)
                            for item in caught))


if __name__ == '__main__':
    unittest.main()
