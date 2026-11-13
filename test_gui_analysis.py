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
from PyQt5.QtCore import QEvent, QPoint, QPointF, QTimer, Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtTest import QTest
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

    def test_peak_cursor_changes_are_used_on_reanalysis_and_can_be_reset(self):
        wavelength = np.linspace(1500, 1600, 401)
        transmission = (1 - 0.2 * np.exp(-((wavelength - 1525) / 2) ** 2)
                        - 0.2 * np.exp(-((wavelength - 1575) / 2) ** 2))
        self.window.T = pd.DataFrame({
            'wavelength_nm': wavelength, 'T_linear': transmission,
            'nu_Hz': analysis.wl2nu(wavelength),
        })
        self.window.param_fields['Param_find_peaks_distance'].setText('50')
        fit_bounds = []

        def fake_fit(nu_res, transmission_res, bounds):
            fit_bounds.append(bounds)
            return (np.array([0.1, 1e9, np.mean(nu_res)]), None,
                    lambda nu, amplitude, width, center: np.ones_like(nu))

        def run_analysis(data, ring, finding, smoothing, fitting, *args, **kwargs):
            peaks, properties = analysis.find_peaks_for_FSR(data, **finding)
            kwargs['peak_result_callback'](data, peaks, properties)
            analysis.resonances_fitting(
                data['wavelength_nm'], data['T_linear'], peaks['idx_peaks'], properties,
                fwhm_bounds_overrides=kwargs['fwhm_bounds_overrides'], **fitting)
            return tuple(object() for _ in range(4))

        with patch.object(gui, 'analysis_main', side_effect=run_analysis):
            with patch('F_LorentzianModel.f_locatized', side_effect=fake_fit):
                self.window.plot_buttons['Show Figure 1'].click()
                resonance = self.window.peak_overlay_items[0]
                resonance.sigPointsClicked.emit(resonance, [resonance.scatter.points()[0]], None)
                self.assertEqual(self.window.selected_peak_index, 100)
                self.assertEqual(len(self.window.peak_cursor_items), 4)
                lower, lower_mirror, upper, upper_mirror = self.window.peak_cursor_items
                center = self.window.T['nu_Hz'].iloc[100]
                lower.setValue(analysis.fwhm_width_to_wavelength_edges(center, 2e10)[1])
                upper.setValue(analysis.fwhm_width_to_wavelength_edges(center, 6e10)[1])
                np.testing.assert_allclose(self.window.manual_fwhm_bounds[100], [2e10, 6e10], rtol=1e-9)
                np.testing.assert_allclose(lower_mirror.value(),
                                           analysis.fwhm_width_to_wavelength_edges(center, 2e10)[0])
                np.testing.assert_allclose(upper_mirror.value(),
                                           analysis.fwhm_width_to_wavelength_edges(center, 6e10)[0])
                self.assertIn('nm', lower.label.format)
                self.window.param_fields['Param_peaks_fitting_Rel_FWHM'].setText('0.25')
                self.window.plot_buttons['Show Figure 1'].click()

        lower_bounds, upper_bounds = fit_bounds[-2]
        np.testing.assert_allclose([lower_bounds[1], upper_bounds[1]], [2e10, 6e10], rtol=1e-9)
        second_lower, second_upper = fit_bounds[-1]
        peak_width = abs(self.window.peak_result_properties['FWHM'][1])
        np.testing.assert_allclose([second_lower[1], second_upper[1]], [0.75 * peak_width, 1.25 * peak_width])
        self.assertIn(100, self.window.manual_fwhm_bounds)
        self.window.button_reset_peak_bounds.click()
        self.assertNotIn(100, self.window.manual_fwhm_bounds)
        self.assertEqual(len(self.window.peak_cursor_items), 4)

    def test_changed_data_or_peak_settings_discard_manual_bounds(self):
        wavelength = np.linspace(1500, 1600, 401)
        self.window.T = pd.DataFrame({
            'wavelength_nm': wavelength, 'T_linear': np.ones(401),
            'nu_Hz': analysis.wl2nu(wavelength),
        })
        self.window.retrieve_params()
        self.window.sync_peak_context()
        self.window.manual_fwhm_bounds[100] = (1e9, 2e9)
        self.window.T.loc[10, 'T_linear'] = 0.8
        self.window.sync_peak_context()
        self.assertEqual(self.window.manual_fwhm_bounds, {})
        self.window.manual_fwhm_bounds[100] = (1e9, 2e9)
        self.window.param_fields['Param_find_peaks_distance'].setText('100')
        self.window.retrieve_params()
        self.window.sync_peak_context()
        self.assertEqual(self.window.manual_fwhm_bounds, {})
        self.assertIn('Manual FWHM bounds cleared', self.window.results_output.toPlainText())

    def test_importing_new_data_discards_manual_bounds(self):
        data = pd.DataFrame({'wavelength_nm': [1500.0, 1501.0], 'T_linear': [1.0, 0.9],
                             'T_dB': [0.0, -0.5], 'nu_Hz': analysis.wl2nu(np.array([1500.0, 1501.0]))})
        self.window.manual_fwhm_bounds[0] = (1e9, 2e9)
        with patch.object(gui, 'load_data', return_value=data):
            self.window.load_csv(file_name='new.csv')
        self.assertEqual(self.window.manual_fwhm_bounds, {})
        self.assertIn('Manual FWHM bounds cleared', self.window.results_output.toPlainText())

    def test_mouse_selects_peak_and_drags_fwhm_cursor(self):
        wavelength = np.linspace(1500, 1600, 401)
        transmission = 1 - 0.2 * np.exp(-((wavelength - 1550) / 2) ** 2)
        data = pd.DataFrame({
            'wavelength_nm': wavelength, 'T_linear': transmission,
            'nu_Hz': analysis.wl2nu(wavelength),
        })
        self.window.T = data
        self.window.retrieve_params()
        self.window.sync_peak_context()
        widget = self.window.plot_widget_operation
        widget.plot(wavelength, transmission)
        peaks, properties = analysis.find_peaks_for_FSR(data, **self.window.Param_find_peaks)
        self.window.show_peak_overlay(data, peaks, properties)
        self.window.show()
        self.app.processEvents()
        widget.setXRange(1546, 1556, padding=0)
        widget.setYRange(0.75, 1.05, padding=0)
        self.app.processEvents()

        view = widget.getViewBox()
        viewport = widget.viewport()
        peak = widget.mapFromScene(view.mapViewToScene(QPointF(1550, 0.8)))
        QTest.mouseClick(viewport, Qt.LeftButton, Qt.NoModifier, peak, 20)
        self.app.processEvents()
        self.assertEqual(self.window.selected_peak_index, 200)
        cursor = self.window.peak_cursor_items[2]
        initial = cursor.value()
        start = widget.mapFromScene(view.mapViewToScene(QPointF(initial, 0.9)))
        end = QPoint(start.x() - 30, start.y())
        QTest.mouseMove(viewport, start, 20)
        self.app.processEvents()
        QTest.mousePress(viewport, Qt.LeftButton, Qt.NoModifier, start, 20)
        move = QMouseEvent(QEvent.MouseMove, QPointF(end), QPointF(viewport.mapToGlobal(end)),
                           Qt.NoButton, Qt.LeftButton, Qt.NoModifier)
        QApplication.sendEvent(viewport, move)
        self.app.processEvents()
        QTest.mouseRelease(viewport, Qt.LeftButton, Qt.NoModifier, end, 20)
        self.app.processEvents()
        self.assertLess(cursor.value(), initial)
        self.assertIn(200, self.window.manual_fwhm_bounds)

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

    def test_failed_analysis_lays_out_only_new_figures_before_showing(self):
        old_figure = plt.figure()
        figures = []
        events = []

        def fail_analysis(*args, **kwargs):
            figures.extend([plt.figure(), plt.figure()])
            raise RuntimeError('analysis failed')

        def layout(figure):
            events.append(('layout', figure))
            if figure is figures[0]:
                raise ValueError('layout failed')

        def show(figure):
            events.append(('show', figure))

        with patch.object(gui, 'analysis_main', side_effect=fail_analysis):
            with patch.object(matplotlib.figure.Figure, 'tight_layout', new=layout):
                with patch.object(matplotlib.figure.Figure, 'show', new=show):
                    with contextlib.redirect_stderr(io.StringIO()):
                        self.window.plot_buttons['Show Figure 1'].click()

        self.assertNotIn(('layout', old_figure), events)
        for figure in figures:
            self.assertLess(events.index(('layout', figure)), events.index(('show', figure)))
        self.assertIn('Could not adjust layout for figure', self.window.results_output.toPlainText())

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

    def test_analysis_fsr_legends_clear_right_axes_and_figure_edges(self):
        self.window.load_csv(file_name='test_W1100_R100um_G400_AfterAnnealing.csv')
        self.window.calculate_difference_yijun()
        with patch.object(plt, 'show'):
            with contextlib.redirect_stdout(io.StringIO()):
                self.window.plot_buttons['Show Figure 1'].click()

        self.assertIn('Analysis completed.', self.window.results_output.toPlainText())
        figure = self.window.Fig_FSR
        axes_by_label = {ax.get_ylabel(): ax for ax in figure.axes if ax.get_ylabel()}
        legends = [ax.get_legend() for ax in figure.axes if not ax.axison]
        self.assertEqual(len(legends), 2)
        expected_labels = [
            ['FSR', 'FSR fitting based on peak fitting', 'FSR not considered',
             'FSR considered', '$n_g$', '$n_g$ fitting based on peak fitting'],
            ['GVD_D', 'GVD D from fitting', r'$\beta_2$', r'$\beta_2$ from fitting'],
        ]
        for legend, expected in zip(legends, expected_labels):
            self.assertEqual([text.get_text() for text in legend.get_texts()], expected)

        for size in ((12, 8.5), (10, 8)):
            with self.subTest(size=size):
                figure.set_size_inches(*size)
                figure.canvas.draw()
                renderer = figure.canvas.get_renderer()
                legend_boxes = [legend.get_window_extent(renderer) for legend in legends]
                for right_label, legend_box in zip(
                    ('$n_g$', r'$\beta_2$ ($ps^2/km$)'), legend_boxes
                ):
                    axis_box = axes_by_label[right_label].yaxis.get_tightbbox(renderer)
                    self.assertGreaterEqual(legend_box.x0 - axis_box.x1, 5)
                    self.assertGreaterEqual(legend_box.x0, figure.bbox.x0)
                    self.assertLessEqual(legend_box.x1, figure.bbox.x1)
                    self.assertGreaterEqual(legend_box.y0, figure.bbox.y0)
                    self.assertLessEqual(legend_box.y1, figure.bbox.y1)
                self.assertFalse(legend_boxes[0].overlaps(legend_boxes[1]))


if __name__ == '__main__':
    unittest.main()
