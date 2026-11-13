import contextlib
import io
import os
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

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
