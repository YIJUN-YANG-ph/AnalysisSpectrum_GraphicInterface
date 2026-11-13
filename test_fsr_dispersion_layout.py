import os
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from PyQt5.QtWidgets import QApplication

import F_FSR_Dispersion as dispersion


class DispersionLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def tearDown(self):
        plt.close('all')

    def test_each_dispersion_figure_is_laid_out_and_updated_after_annotation(self):
        freq = np.linspace(190e12, 195e12, 21)
        fsr = np.full(freq.shape, 2e11)
        fit = np.poly1d([2e11])
        polynomial = np.poly1d([1e4, 1e6, 1e8, 0])
        calls = []
        original = Figure.tight_layout

        def record_layout(figure, *args, **kwargs):
            calls.append(figure)
            return original(figure, *args, **kwargs)

        with patch.object(Figure, 'tight_layout', new=record_layout):
            with patch('F_fit_fsr_polynomial_robust.fit_fsr_polynomial_robust',
                       side_effect=[(fit, np.ones(freq.shape, dtype=bool)),
                                    (polynomial, np.ones(freq.shape, dtype=bool))]):
                f_fsr, mu, dint, handles = dispersion.F_FSR_intDisp(freq, fsr)
                self.assertIn(handles['fig_fsr'], calls)
                self.assertIn(handles['fig_disp'], calls)

                unrelated = plt.figure()
                calls.clear()
                with patch.object(plt, 'show'):
                    dispersion.F_intDisp_HOD(mu, dint, f_fsr, freq, plot_handles=handles)
                self.assertEqual(calls, [handles['fig_disp']])
                self.assertNotIn(unrelated, calls)
                self.assertTrue(handles['ax_disp'].texts)

    def test_no_plot_does_not_adjust_any_figure(self):
        freq = np.linspace(190e12, 195e12, 21)
        fsr = np.full(freq.shape, 2e11)
        with patch.object(Figure, 'tight_layout') as layout:
            with patch('F_fit_fsr_polynomial_robust.fit_fsr_polynomial_robust',
                       return_value=(np.poly1d([2e11]), np.ones(freq.shape, dtype=bool))):
                dispersion.F_FSR_intDisp(freq, fsr, isPlot=False)
        layout.assert_not_called()


if __name__ == '__main__':
    unittest.main()
