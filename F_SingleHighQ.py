import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, 
    QVBoxLayout, QHBoxLayout, QPushButton, 
    QFileDialog, QFormLayout, QDoubleSpinBox, QLabel
)
from scipy.optimize import curve_fit
# import partial
from functools import partial
from F_ConvertUnits import nu2wl, wl2nu, dB2linear
from F_LoadData import load_data 
# 1. Define the Lorentzian Model
def lorentzian(x, amplitude, center, fwhm, offset):
    """
    Defines the Lorentzian function.
    x: independent variable
    amplitude: peak height
    center: peak center (x0)
    fwhm: full width at half-maximum
    offset: y-baseline
    """
    # Using half-width at half-maximum (gamma) for simpler formula
    gamma = fwhm / 2.0
    return offset + amplitude * (gamma**2 / ((x - center)**2 + gamma**2))

# 2. Define the Main Application Window
class LorentzianFitter(QMainWindow):
    def __init__(self):
        super().__init__()

        # self.load_data = partial(load_data(file_name='D80-G400-W1598.679-0-1.5(241025).txt',
        #                           wl_name='frequency',data_name='power1'))
        # load_data_singleQ = partial(load_data,wl_name='frequency',data_name='power1')


        self.setWindowTitle("Interactive Lorentzian Fitter")
        self.setGeometry(100, 100, 1000, 600)

        # --- Main Layout ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        # Use a horizontal layout for controls | plot
        self.main_layout = QHBoxLayout(self.central_widget)

        # --- Control Panel ---
        self.control_panel = QWidget()
        self.control_layout = QVBoxLayout()
        self.control_panel.setLayout(self.control_layout)
        
        # File loading
        self.btn_load = QPushButton("Load Data (.txt or .csv)")
        
        self.btn_load.clicked.connect(self.load_data_singleQ)
        self.control_layout.addWidget(self.btn_load)

        # Parameter Input Form
        self.form_layout = QFormLayout()
        
        # Create QDoubleSpinBox widgets for parameters
        self.spin_amplitude = QDoubleSpinBox(self)
        self.spin_center = QDoubleSpinBox(self)
        self.spin_fwhm = QDoubleSpinBox(self)
        self.spin_offset = QDoubleSpinBox(self)
        
        self.param_widgets = [self.spin_amplitude, self.spin_center, self.spin_fwhm, self.spin_offset]

        # Set realistic ranges (adjust as needed for your data)
        for spin in self.param_widgets:
            spin.setRange(-1e9, 1e9)
            spin.setDecimals(3)
            spin.setSingleStep(0.1)
            # Connect the valueChanged signal to update the guess
            spin.valueChanged.connect(self.update_guess_plot)
            
        self.spin_amplitude.setValue(1.0)
        self.spin_fwhm.setValue(0.1)

        self.form_layout.addRow("Amplitude:", self.spin_amplitude)
        self.form_layout.addRow("Center (x₀):", self.spin_center)
        self.form_layout.addRow("FWHM (w):", self.spin_fwhm)
        self.form_layout.addRow("Offset (c):", self.spin_offset)
        
        self.control_layout.addLayout(self.form_layout)

        # Fit button
        self.btn_fit = QPushButton("Run Fit")
        self.btn_fit.clicked.connect(self.run_fit)
        self.control_layout.addWidget(self.btn_fit)
        
        # Labels for fit results
        self.control_layout.addWidget(QLabel("--- Fit Results ---"))
        self.result_layout = QFormLayout()
        self.res_amplitude = QLabel("-")
        self.res_center = QLabel("-")
        self.res_fwhm = QLabel("-")
        self.res_offset = QLabel("-")
        self.result_layout.addRow("Amplitude:", self.res_amplitude)
        self.result_layout.addRow("Center:", self.res_center)
        self.result_layout.addRow("FWHM:", self.res_fwhm)
        self.result_layout.addRow("Offset:", self.res_offset)
        self.control_layout.addLayout(self.result_layout)

        self.control_layout.addStretch() # Push everything to the top
        self.main_layout.addWidget(self.control_panel)

        # --- Plotting Area ---
        self.plot_widget = pg.PlotWidget()
        self.main_layout.addWidget(self.plot_widget, stretch=3) # Give plot more space
        # Add legend
        self.plot_widget.addLegend()
        # Second plot widget for better 
        self.plot_widget_linear = pg.PlotWidget()
        self.plot_widget_linear.setBackground('w')
        self.plot_widget_linear.setLabel('left', 'Intensity (a.u.)')
        self.plot_widget_linear.setLabel('bottom', 'Frequency (Hz)')
        # Add legend
        self.plot_widget_linear.addLegend()

        # Create plot items (we'll update these)
        self.plot_data = self.plot_widget.plot(
            [], [], pen=None, symbol='o', symbolSize=5, symbolBrush=(100, 100, 150), 
            name="Raw Data"
        )
        # Second plot item for linear view
        self.plot_data_linear = self.plot_widget_linear.plot(
            [], [], pen=None, symbol='o', symbolSize=5, symbolBrush=(100, 100, 150), 
            name="Raw Data (Linear)"
        )
        
        self.plot_guess = self.plot_widget.plot(
            [], [], pen=pg.mkPen('r', style=pg.QtCore.Qt.PenStyle.DashLine, width=2),
            name="Initial Guess"
        )
        self.plot_fit = self.plot_widget.plot(
            [], [], pen=pg.mkPen('g', width=3),
            name="Fitted Curve"
        )

        # --- Class variables to store data ---
        self.x_data = None
        self.y_data = None

    # load_data = partial(load_data(file_name='D80-G400-W1598.679-0-1.5(241025).txt',
    #                               wl_name='frequency',data_name='power1'))
    # load_data_singleQ = partial(load_data,wl_name='frequency',data_name='power1')
    def load_data_singleQ(self):
        """
        Opens a file dialog to load a .txt or .csv file.
        Assumes 2-column data (x, y) separated by comma, space, or tab.
        """
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "", "Text Files (*.txt *.csv)"
        )
        if file_name:
            try:                
                # file_name = r'D80-G400-W1598.679-0-1.5(241025).txt'
                T = load_data(file_name, wl_name='frequency', data_name='power1')
                #change the key names according to the actual file
                T = T.rename(columns={'wavelength_nm':'nu_Hz', 'nu_Hz':'wavelength_nm',})
                # convert nu_Hz from THz to Hz
                T['nu_Hz'] = T['nu_Hz'] * 1e12
                T['wavelength_nm'] = nu2wl(T['nu_Hz']) # convert to nm
                import matplotlib.pyplot as plt
                plt.plot(T['nu_Hz'], T['T_dB'])
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Electrical Power (dB)')
                print(T.head())
                self.x_data = T['nu_Hz'].values
                self.y_data = T['T_dB'].values

                # data = np.loadtxt(file_name, delimiter=None, skiprows=1) # Assumes header, change skiprows=0 if no header

                # self.x_data = data[:, 1]
                # self.y_data = data[:, 2]
                
                # Plot the raw data
                self.plot_data.setData(self.x_data, self.y_data)
                # Also plot on the linear scale plot
                y_data_linear = T['T_linear'].values
                self.plot_data_linear.setData(self.x_data, y_data_linear)
                '''
                # Make intelligent guesses for initial parameters
                self.spin_offset.setValue(np.min(self.y_data))
                self.spin_amplitude.setValue(np.max(self.y_data) - np.min(self.y_data))
                self.spin_center.setValue(self.x_data[np.argmax(self.y_data)])
                self.spin_fwhm.setValue((self.x_data.max() - self.x_data.min()) * 0.1) # Guess 10% of range
                
                # Clear old fit plot
                self.plot_fit.setData([], [])
                '''
            except Exception as e:
                print(f"Error loading file: {e}")

    def update_guess_plot(self):
        """
        Reads parameters from the spin boxes and plots the "guess" curve.
        """
        if self.x_data is None:
            return

        # Get current parameters from the GUI
        amplitude = self.spin_amplitude.value()
        center = self.spin_center.value()
        fwhm = self.spin_fwhm.value()
        offset = self.spin_offset.value()

        # Generate the guess curve
        y_guess = lorentzian(self.x_data, amplitude, center, fwhm, offset)
        
        # Update the plot
        self.plot_guess.setData(self.x_data, y_guess)

    def run_fit(self):
        """
        Performs the Lorentzian fit using scipy.optimize.curve_fit
        """
        if self.x_data is None:
            return

        # Get initial guesses (p0) from the spin boxes
        p0 = [
            self.spin_amplitude.value(),
            self.spin_center.value(),
            self.spin_fwhm.value(),
            self.spin_offset.value()
        ]
        
        try:
            # Perform the fit
            popt, pcov = curve_fit(lorentzian, self.x_data, self.y_data, p0=p0)
            
            # Get the optimized parameters
            fit_amp, fit_cen, fit_fwhm, fit_off = popt
            
            # Get the standard deviation errors (sqrt of diagonal of covariance matrix)
            perr = np.sqrt(np.diag(pcov))

            # Generate the final fitted curve
            y_fit = lorentzian(self.x_data, *popt)
            
            # Plot the fitted curve
            self.plot_fit.setData(self.x_data, y_fit)
            
            # Update the result labels
            self.res_amplitude.setText(f"{fit_amp:.3f} ± {perr[0]:.3f}")
            self.res_center.setText(f"{fit_cen:.3f} ± {perr[1]:.3f}")
            self.res_fwhm.setText(f"{fit_fwhm:.3f} ± {perr[2]:.3f}")
            self.res_offset.setText(f"{fit_off:.3f} ± {perr[3]:.3f}")

        except RuntimeError as e:
            print(f"Fit failed: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

# 3. Run the Application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = LorentzianFitter()
    main_window.show()
    sys.exit(app.exec())