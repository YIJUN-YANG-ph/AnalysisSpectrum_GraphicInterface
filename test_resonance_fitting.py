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
import F_Bounds


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

    def test_fwhm_bounds_are_calculated_before_calling_bounds_helper(self):
        wavelength = pd.Series(np.linspace(1500, 1560, 301))
        transmission = pd.Series(np.full(len(wavelength), 0.9))
        indices = np.array([100, 200])
        properties = {'prominences': np.ones(2), 'FWHM': np.array([-2e9, 4e9])}

        with patch.object(F_Bounds, 'F_Bounds_SingleLorentzian',
                          wraps=F_Bounds.F_Bounds_SingleLorentzian) as make_bounds:
            with patch('F_LorentzianModel.f_locatized', side_effect=self.fake_fit(None)) as fit:
                analysis.resonances_fitting(wavelength, transmission, indices, properties)

        self.assertEqual(make_bounds.call_count, 2)
        for call, fit_call, peak_width in zip(make_bounds.call_args_list, fit.call_args_list, (2e9, 4e9)):
            np.testing.assert_allclose(call.kwargs['FWHM_lower'], peak_width * 0.01)
            np.testing.assert_allclose(call.kwargs['FWHM_upper'], peak_width * 1.99)
            self.assertNotIn('FWHM0', call.kwargs)
            self.assertNotIn('Rel_FWHM', call.kwargs)
            lower, upper = fit_call.args[2]
            np.testing.assert_allclose([lower[1], upper[1]], [peak_width * 0.01, peak_width * 1.99])

    def test_custom_relative_fwhm_controls_fitting_bounds(self):
        wavelength = pd.Series(np.linspace(1500, 1560, 301))
        transmission = pd.Series(np.full(len(wavelength), 0.9))
        properties = {'prominences': np.array([0.1]), 'FWHM': np.array([2e9])}
        with patch('F_LorentzianModel.f_locatized', side_effect=self.fake_fit(None)) as fit:
            analysis.resonances_fitting(
                wavelength, transmission, [100], properties, Rel_FWHM=0.25)
        lower, upper = fit.call_args.args[2]
        np.testing.assert_allclose([lower[1], upper[1]], [1.5e9, 2.5e9])

    def test_invalid_relative_fwhm_is_rejected_before_fitting(self):
        for value in (0, -0.1, 1.1, float('nan'), float('inf')):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'Rel_FWHM'):
                    analysis.resonances_fitting(None, None, [], {}, Rel_FWHM=value)

    def test_wavelength_cursor_conversion_round_trips_frequency_widths(self):
        center = analysis.wl2nu(1550.0)
        for width in (0, 1e9, 20e9, 2e12):
            with self.subTest(width=width):
                short, long = analysis.fwhm_width_to_wavelength_edges(center, width)
                self.assertLessEqual(short, analysis.nu2wl(center))
                self.assertGreaterEqual(long, analysis.nu2wl(center))
                np.testing.assert_allclose(
                    analysis.fwhm_width_from_long_wavelength(center, long),
                    width, rtol=1e-10, atol=0.1)
                np.testing.assert_allclose(long - short,
                                           analysis.nu2wl(center - width / 2)
                                           - analysis.nu2wl(center + width / 2))
        with self.assertRaises(ValueError):
            analysis.fwhm_width_to_wavelength_edges(center, 2 * center)
        with self.assertRaises(ValueError):
            analysis.fwhm_width_from_long_wavelength(center, 1500)

    def test_manual_bounds_override_only_the_matching_peak(self):
        wavelength = pd.Series(np.linspace(1500, 1560, 301))
        transmission = pd.Series(np.full(len(wavelength), 0.9))
        properties = {'prominences': np.ones(2), 'FWHM': np.array([2e9, 4e9])}
        with patch('F_LorentzianModel.f_locatized', side_effect=self.fake_fit(None)) as fit:
            analysis.resonances_fitting(
                wavelength, transmission, [100, 200], properties, Rel_FWHM=0.25,
                fwhm_bounds_overrides={100: (0.4e9, 3e9)})
        first_lower, first_upper = fit.call_args_list[0].args[2]
        second_lower, second_upper = fit.call_args_list[1].args[2]
        np.testing.assert_allclose([first_lower[1], first_upper[1]], [0.4e9, 3e9])
        np.testing.assert_allclose([second_lower[1], second_upper[1]], [3e9, 5e9])

    def test_invalid_manual_bounds_are_rejected_before_fitting(self):
        for bounds in ((-1, 2), (1, 1), (2, 1), (0, float('inf')), (float('nan'), 2)):
            with self.subTest(bounds=bounds):
                with self.assertRaisesRegex(ValueError, 'FWHM bounds'):
                    analysis.resonances_fitting(None, None, [], {}, fwhm_bounds_overrides={0: bounds})

    def test_analysis_passes_manual_bounds_to_resonance_fitting(self):
        wavelength = pd.Series(np.linspace(1500, 1560, 2201))
        transmission = pd.Series(np.full(len(wavelength), 0.9))
        indices = np.arange(50, 2151, 50)
        properties = {'prominences': np.ones(len(indices)),
                      'FWHM': np.full(len(indices), 1e9)}
        data = pd.DataFrame({'wavelength_nm': wavelength, 'T_linear': transmission,
                             'nu_Hz': analysis.wl2nu(wavelength)})
        overrides = {50: (1e8, 2e9)}
        with patch.object(analysis, 'find_peaks_for_FSR',
                          return_value=(pd.DataFrame({'idx_peaks': indices}), properties)):
            with patch.object(analysis, 'RemoveOffset_Savgol',
                              return_value=(transmission, np.ones(len(wavelength)))):
                with patch.object(analysis, 'resonances_fitting',
                                  side_effect=RuntimeError('stop at fitting')) as fit:
                    with contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaisesRegex(RuntimeError, 'stop at fitting'):
                            analysis.analysis_main(
                                data, {'diameter': 200}, {}, {}, {}, {}, {},
                                {'pump wl': 1550}, fwhm_bounds_overrides=overrides)
        self.assertIs(fit.call_args.kwargs['fwhm_bounds_overrides'], overrides)

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
