import contextlib
import io
import os
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

import components_AnalysisSpectrum as analysis
import GUI_main_analysis as gui


class AnalysisErrorDisplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        plt.ioff()
        with patch.object(gui.MainWindow, 'load_csv'):
            self.window = gui.MainWindow()

    def tearDown(self):
        plt.close('all')
        self.window.close()

    def test_plot_axis_labels_remain_after_clearing(self):
        for widget, ylabel in (
            (self.window.plot_widget, 'data (dB)'),
            (self.window.plot_widget_operation, 'data (a.u.)'),
        ):
            with self.subTest(ylabel=ylabel):
                widget.clear()
                self.assertEqual(widget.getAxis('bottom').labelText, 'wavelength (nm)')
                self.assertEqual(widget.getAxis('left').labelText, ylabel)

    def test_relative_fwhm_is_saved_loaded_and_passed_to_analysis(self):
        field = self.window.param_fields['Param_peaks_fitting_Rel_FWHM']
        self.assertEqual(field.text(), '0.99')
        field.setText('0.25')
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, 'parameters.txt')
            with patch.object(gui.QFileDialog, 'getSaveFileName', return_value=(filename, '')):
                self.window.save_current_params()
            with open(filename) as saved:
                self.assertIn('Param_peaks_fitting_Rel_FWHM: 0.25\n', saved.read())
            field.setText('0.5')
            with patch.object(gui.QFileDialog, 'getOpenFileName', return_value=(filename, '')):
                self.window.load_params_from_file()
            self.assertEqual(field.text(), '0.25')

        with patch.object(gui, 'analysis_main', return_value=tuple(object() for _ in range(4))) as run:
            self.window.plot_buttons['Show Figure 1'].click()
        self.assertEqual(run.call_args.args[4]['Rel_FWHM'], 0.25)

    def test_old_parameter_file_preserves_default_relative_fwhm(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, 'old_parameters.txt')
            with open(filename, 'w') as saved:
                saved.write('Param_peaks_fitting_A_margin: 0.03\n')
            with patch.object(gui.QFileDialog, 'getOpenFileName', return_value=(filename, '')):
                self.window.load_params_from_file()
        self.window.retrieve_params()
        self.assertEqual(self.window.Param_peaks_fitting['Rel_FWHM'], 0.99)
        self.assertEqual(self.window.Param_peaks_fitting['A_margin'], 0.03)

    def test_invalid_relative_fwhm_prevents_analysis(self):
        for value in ('0', '-0.1', '1.1', 'nan', 'inf', 'invalid'):
            with self.subTest(value=value):
                self.window.param_fields['Param_peaks_fitting_Rel_FWHM'].setText(value)
                with patch.object(gui, 'analysis_main') as run:
                    with contextlib.redirect_stderr(io.StringIO()):
                        self.window.plot_buttons['Show Figure 1'].click()
                run.assert_not_called()
                self.assertIn('ValueError: Rel_FWHM', self.window.results_output.toPlainText())

    def test_analyze_overlays_peak_geometry_and_replaces_it_on_reanalysis(self):
        wavelength = np.linspace(1500, 1600, 401)
        transmission = 1 - 0.2 * np.exp(-((wavelength - 1550) / 2) ** 2)
        data = pd.DataFrame({
            'wavelength_nm': wavelength,
            'T_linear': transmission,
            'nu_Hz': analysis.wl2nu(wavelength),
        })
        self.window.T = data
        curve = self.window.plot_widget_operation.plot(wavelength, transmission)

        with patch.object(analysis, 'calcul_FSR', side_effect=RuntimeError('stop after peaks')):
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                self.window.plot_buttons['Show Figure 1'].click()
                old_items = list(self.window.peak_overlay_items)
                self.window.plot_buttons['Show Figure 1'].click()

        self.assertEqual(len(self.window.peak_overlay_items), 3)
        self.assertTrue(all(item not in self.window.plot_widget_operation.listDataItems()
                            for item in old_items))
        self.assertIn(curve, self.window.plot_widget_operation.listDataItems())
        resonance, prominence, fwhm = self.window.peak_overlay_items
        np.testing.assert_array_equal(resonance.getData()[0], [1550])
        np.testing.assert_array_equal(resonance.getData()[1], [transmission[200]])

        figure = plt.figure(max(plt.get_fignums()))
        vertical, horizontal = figure.axes[0].collections[:2]
        for item, segments in ((prominence, vertical.get_segments()),
                               (fwhm, horizontal.get_segments())):
            x, y = item.getData()
            np.testing.assert_allclose(np.column_stack((x, y)), np.asarray(segments).reshape(-1, 2))
        self.assertIn('RuntimeError: stop after peaks', self.window.results_output.toPlainText())

        self.window.plot_buttons['Show Figure 3'].click()
        self.assertEqual(self.window.peak_overlay_items, [])
        self.assertEqual(self.window.plot_widget_operation.listDataItems(), [])

    def test_empty_peak_results_remove_previous_overlay(self):
        data = pd.DataFrame({'wavelength_nm': [1500, 1501], 'T_linear': [1, 1]})
        self.window.show_peak_overlay(
            data, pd.DataFrame({'idx_peaks': [0]}),
            {'prominences': np.array([0.1]), 'FWHM_left_ips': np.array([0.0]),
             'FWHM_right_ips': np.array([1.0]), 'FWHM_width_heights': np.array([1.05])})
        self.window.show_peak_overlay(data, pd.DataFrame({'idx_peaks': []}), {})
        self.assertEqual(self.window.peak_overlay_items, [])
        self.assertEqual(self.window.plot_widget_operation.listDataItems(), [])

    def test_analysis_error_shows_only_new_figures_and_full_traceback(self):
        old_figure = plt.figure()
        new_figures = []

        def fail_analysis(*args, **kwargs):
            figure, axis = plt.subplots()
            axis.plot([1, 2], [3, 4])
            new_figures.append(figure)
            raise RuntimeError('simulated analysis failure')

        stderr = io.StringIO()
        with patch.object(gui, 'analysis_main', side_effect=fail_analysis):
            with contextlib.redirect_stderr(stderr), patch('sys.excepthook'):
                self.window.plot_buttons['Show Figure 1'].click()

        output = self.window.results_output.toPlainText()
        self.assertIn('Traceback (most recent call last):', output)
        self.assertIn('RuntimeError: simulated analysis failure', output)
        self.assertIn(stderr.getvalue().strip(), output)
        self.assertNotIn('Analysis completed.', output)
        self.assertFalse(old_figure.canvas.manager.window.isVisible())
        self.assertTrue(new_figures[0].canvas.manager.window.isVisible())
        self.assertEqual(len(new_figures[0].axes[0].lines), 1)

        events = []
        QTimer.singleShot(0, lambda: events.append('processed'))
        self.app.processEvents()
        self.assertEqual(events, ['processed'])

    def test_error_showing_one_figure_does_not_hide_other_figures(self):
        figures = []

        def fail_analysis(*args, **kwargs):
            figures.extend([plt.figure(), plt.figure()])
            raise RuntimeError('analysis failed')

        stderr = io.StringIO()
        with patch.object(gui, 'analysis_main', side_effect=fail_analysis):
            with patch.object(matplotlib.figure.Figure, 'show',
                              side_effect=[RuntimeError('display failed'), None]) as show:
                with contextlib.redirect_stderr(stderr):
                    self.window.plot_buttons['Show Figure 1'].click()

        self.assertEqual(show.call_count, 2)
        self.assertIn('RuntimeError: analysis failed', stderr.getvalue())
        self.assertIn('RuntimeError: display failed', stderr.getvalue())
        self.assertIn('Could not display figure', self.window.results_output.toPlainText())

    def test_invalid_parameters_report_traceback_without_running_analysis(self):
        self.window.param_fields['Param_Read_csv_skiprows'].setText('-1')
        stderr = io.StringIO()
        with patch.object(gui, 'analysis_main') as analysis:
            with contextlib.redirect_stderr(stderr):
                self.window.plot_buttons['Show Figure 1'].click()

        analysis.assert_not_called()
        self.assertIn('ValueError: Read csv skiprows must be a non-negative integer.',
                      self.window.results_output.toPlainText())
        self.assertIn('Traceback (most recent call last):', stderr.getvalue())

    def test_successful_analysis_keeps_existing_result_handling(self):
        figures = tuple(object() for _ in range(4))
        with patch.object(gui, 'analysis_main', return_value=figures):
            self.window.plot_buttons['Show Figure 1'].click()

        self.assertEqual(
            (self.window.Fig_FSR, self.window.Fig_peak_fitting,
             self.window.Fig_resonance, self.window.fig_loss),
            figures,
        )
        self.assertIn('Analysis completed.', self.window.results_output.toPlainText())


if __name__ == '__main__':
    unittest.main()
