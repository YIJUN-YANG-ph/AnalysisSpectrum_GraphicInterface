# Create a gui for single high Q analysis
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, 
    QVBoxLayout, QHBoxLayout, QPushButton, 
    QFileDialog, QFormLayout, QDoubleSpinBox, QLabel,
    QSpinBox, QGroupBox, QScrollArea, QTextEdit, QTabWidget,
    QLineEdit, QComboBox
)
from PyQt5.QtCore import Qt

# Import the SingleHighQ class for analysis
from C_SingleHighQ import SingleHighQ

###########################################################################################
# Some default parameters for analysis
param_find_peaks = {'distance':400,# in number of points
                        'prominence':0.0001,# need to be adjusted according to the actual data
                        'width':5,# in number of points
                        'rel_height':1,# don't change this one
                        }

param_find_DoubletResonance = {'distance':2,
                                'prominence':0.08,
                                'width':2,
                                'rel_height':0.5,
                                } # don't add random stuff here, only the parameters suitable for scipy.find_peaks
    
param_rel_Doublet = {'Rel_A1':0.99,
                        'Rel_A2':0.99,
                    #  'Rel_FWHM1':0.8,
                    #  'Rel_FWHM2':0.8,
                        'Rel_Peak1': 0.8,
                        'Rel_Peak2':0.8,
                        '1st peak':0, # multiple peaks found, choose which the one as the first peak
                        '2nd peak':1,}# multiple peaks found, choose which the one as the second peak

param_rel_Singlet = {'Rel_A':0.2,
                     'Rel_FWHM':0.2,
                     'Rel_Peak':0.2,
                     'factor_remove_points':1,}
############################################################################################
class GUI_SingleHighQ(QMainWindow):
    """GUI for single high Q resonance analysis using the SingleHighQ class"""
    def __init__(self):
        super().__init__()
    
        self.setWindowTitle("Single High Q Resonance Analysis")
        self.setGeometry(100, 100, 1400, 800)

        # Initialize data storage
        self.analyzer = None
        self.data_file = None
        self.offset_file = None
        self.T_Data = None
        self.T_Offset = None
        self.T_corrected = None
        self.T_normalised = None

        # --- Main Layout ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        # Use a horizontal layout for controls | plot
        self.main_layout = QHBoxLayout(self.central_widget)

        # --- Control Panel (Left Side) ---
        self.setup_control_panel()
        
        # --- Plot Panel (Right Side) ---
        self.setup_plot_panel()
        
        # Add panels to main layout
        self.main_layout.addWidget(self.control_scroll, 1)  # 1 part for controls
        self.main_layout.addWidget(self.plot_tabs, 2)  # 2 parts for plots

    def setup_control_panel(self):
        """Setup the left control panel with file loading and parameters"""
        # Create scrollable area for controls
        self.control_scroll = QScrollArea()
        self.control_scroll.setWidgetResizable(True)
        self.control_scroll.setMinimumWidth(400)
        
        self.control_panel = QWidget()
        self.control_layout = QVBoxLayout()
        self.control_panel.setLayout(self.control_layout)
        
        # Title
        title_label = QLabel("<h2>Single High Q Analysis</h2>")
        self.control_layout.addWidget(title_label)
        
        # === File Loading Section ===
        file_group = QGroupBox("File Loading")
        file_layout = QVBoxLayout()
        
        # Data file selection
        data_file_layout = QHBoxLayout()
        self.btn_load_data = QPushButton("Load Data File")
        self.btn_load_data.clicked.connect(self.load_data_file)
        self.data_file_label = QLabel("No file loaded")
        self.data_file_label.setWordWrap(True)
        data_file_layout.addWidget(self.btn_load_data)
        data_file_layout.addWidget(self.data_file_label, 1)
        file_layout.addLayout(data_file_layout)
        
        # Offset file selection (optional)
        offset_file_layout = QHBoxLayout()
        self.btn_load_offset = QPushButton("Load Offset File (Optional)")
        self.btn_load_offset.clicked.connect(self.load_offset_file)
        self.offset_file_label = QLabel("No offset file")
        self.offset_file_label.setWordWrap(True)
        offset_file_layout.addWidget(self.btn_load_offset)
        offset_file_layout.addWidget(self.offset_file_label, 1)
        file_layout.addLayout(offset_file_layout)
        
        # Data model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Data Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(['Heterodyne', 'FineScan', 'Default'])
        model_layout.addWidget(self.model_combo, 1)
        file_layout.addLayout(model_layout)
        
        # Center wavelength
        wl_layout = QHBoxLayout()
        wl_layout.addWidget(QLabel("Center λ (nm):"))
        self.wl_spinbox = QDoubleSpinBox()
        self.wl_spinbox.setRange(1000, 2000)
        self.wl_spinbox.setValue(1550)
        self.wl_spinbox.setDecimals(3)
        wl_layout.addWidget(self.wl_spinbox, 1)
        file_layout.addLayout(wl_layout)
        
        file_group.setLayout(file_layout)
        self.control_layout.addWidget(file_group)
        
        # === Peak Finding Parameters ===
        peak_group = QGroupBox("Peak Finding Parameters")
        peak_layout = QFormLayout()
        
        self.peak_distance = QSpinBox()
        self.peak_distance.setRange(1, 10000)
        self.peak_distance.setValue(param_find_peaks['distance'])
        peak_layout.addRow("Distance (points):", self.peak_distance)
        
        self.peak_prominence = QDoubleSpinBox()
        self.peak_prominence.setRange(0.0, 1.0)
        self.peak_prominence.setValue(param_find_peaks['prominence'])
        self.peak_prominence.setDecimals(6)
        self.peak_prominence.setSingleStep(0.0001)
        peak_layout.addRow("Prominence:", self.peak_prominence)
        
        self.peak_width = QSpinBox()
        self.peak_width.setRange(1, 1000)
        self.peak_width.setValue(param_find_peaks['width'])
        peak_layout.addRow("Width (points):", self.peak_width)
        
        self.peak_rel_height = QDoubleSpinBox()
        self.peak_rel_height.setRange(0.0, 1.0)
        self.peak_rel_height.setValue(param_find_peaks['rel_height'])
        self.peak_rel_height.setDecimals(2)
        peak_layout.addRow("Rel Height:", self.peak_rel_height)
        
        peak_group.setLayout(peak_layout)
        self.control_layout.addWidget(peak_group)
        
        # === Singlet Fitting Parameters ===
        singlet_group = QGroupBox("Singlet Fitting Parameters")
        singlet_layout = QFormLayout()
        
        self.singlet_rel_a = QDoubleSpinBox()
        self.singlet_rel_a.setRange(0.0, 2.0)
        self.singlet_rel_a.setValue(param_rel_Singlet['Rel_A'])
        self.singlet_rel_a.setDecimals(3)
        self.singlet_rel_a.setSingleStep(0.1)
        singlet_layout.addRow("Rel_A:", self.singlet_rel_a)
        
        self.singlet_rel_fwhm = QDoubleSpinBox()
        self.singlet_rel_fwhm.setRange(0.0, 2.0)
        self.singlet_rel_fwhm.setValue(param_rel_Singlet['Rel_FWHM'])
        self.singlet_rel_fwhm.setDecimals(3)
        self.singlet_rel_fwhm.setSingleStep(0.1)
        singlet_layout.addRow("Rel_FWHM:", self.singlet_rel_fwhm)
        
        self.singlet_rel_peak = QDoubleSpinBox()
        self.singlet_rel_peak.setRange(0.0, 2.0)
        self.singlet_rel_peak.setValue(param_rel_Singlet['Rel_Peak'])
        self.singlet_rel_peak.setDecimals(3)
        self.singlet_rel_peak.setSingleStep(0.1)
        singlet_layout.addRow("Rel_Peak:", self.singlet_rel_peak)
        
        singlet_group.setLayout(singlet_layout)
        self.control_layout.addWidget(singlet_group)
        
        # === Doublet Resonance Parameters ===
        doublet_group = QGroupBox("Doublet Resonance Parameters")
        doublet_layout = QFormLayout()
        
        self.doublet_distance = QSpinBox()
        self.doublet_distance.setRange(1, 1000)
        self.doublet_distance.setValue(param_find_DoubletResonance['distance'])
        doublet_layout.addRow("Distance:", self.doublet_distance)
        
        self.doublet_prominence = QDoubleSpinBox()
        self.doublet_prominence.setRange(0.0, 1.0)
        self.doublet_prominence.setValue(param_find_DoubletResonance['prominence'])
        self.doublet_prominence.setDecimals(3)
        self.doublet_prominence.setSingleStep(0.01)
        doublet_layout.addRow("Prominence:", self.doublet_prominence)
        
        self.doublet_width = QSpinBox()
        self.doublet_width.setRange(1, 100)
        self.doublet_width.setValue(param_find_DoubletResonance['width'])
        doublet_layout.addRow("Width:", self.doublet_width)
        
        self.doublet_rel_height = QDoubleSpinBox()
        self.doublet_rel_height.setRange(0.0, 1.0)
        self.doublet_rel_height.setValue(param_find_DoubletResonance['rel_height'])
        self.doublet_rel_height.setDecimals(2)
        doublet_layout.addRow("Rel Height:", self.doublet_rel_height)
        
        doublet_group.setLayout(doublet_layout)
        self.control_layout.addWidget(doublet_group)
        
        # === Doublet Relative Parameters ===
        doublet_rel_group = QGroupBox("Doublet Relative Parameters")
        doublet_rel_layout = QFormLayout()
        
        self.doublet_rel_a1 = QDoubleSpinBox()
        self.doublet_rel_a1.setRange(0.0, 2.0)
        self.doublet_rel_a1.setValue(param_rel_Doublet['Rel_A1'])
        self.doublet_rel_a1.setDecimals(3)
        self.doublet_rel_a1.setSingleStep(0.1)
        doublet_rel_layout.addRow("Rel_A1:", self.doublet_rel_a1)
        
        self.doublet_rel_a2 = QDoubleSpinBox()
        self.doublet_rel_a2.setRange(0.0, 2.0)
        self.doublet_rel_a2.setValue(param_rel_Doublet['Rel_A2'])
        self.doublet_rel_a2.setDecimals(3)
        self.doublet_rel_a2.setSingleStep(0.1)
        doublet_rel_layout.addRow("Rel_A2:", self.doublet_rel_a2)
        
        self.doublet_rel_peak1 = QDoubleSpinBox()
        self.doublet_rel_peak1.setRange(0.0, 2.0)
        self.doublet_rel_peak1.setValue(param_rel_Doublet['Rel_Peak1'])
        self.doublet_rel_peak1.setDecimals(3)
        self.doublet_rel_peak1.setSingleStep(0.1)
        doublet_rel_layout.addRow("Rel_Peak1:", self.doublet_rel_peak1)
        
        self.doublet_rel_peak2 = QDoubleSpinBox()
        self.doublet_rel_peak2.setRange(0.0, 2.0)
        self.doublet_rel_peak2.setValue(param_rel_Doublet['Rel_Peak2'])
        self.doublet_rel_peak2.setDecimals(3)
        self.doublet_rel_peak2.setSingleStep(0.1)
        doublet_rel_layout.addRow("Rel_Peak2:", self.doublet_rel_peak2)
        
        self.doublet_1st_peak = QSpinBox()
        self.doublet_1st_peak.setRange(0, 10)
        self.doublet_1st_peak.setValue(param_rel_Doublet['1st peak'])
        doublet_rel_layout.addRow("1st Peak Index:", self.doublet_1st_peak)
        
        self.doublet_2nd_peak = QSpinBox()
        self.doublet_2nd_peak.setRange(0, 10)
        self.doublet_2nd_peak.setValue(param_rel_Doublet['2nd peak'])
        doublet_rel_layout.addRow("2nd Peak Index:", self.doublet_2nd_peak)
        
        doublet_rel_group.setLayout(doublet_rel_layout)
        self.control_layout.addWidget(doublet_rel_group)
        
        # === Analysis Buttons ===
        button_layout = QVBoxLayout()
        
        self.btn_run_singlet = QPushButton("Run Singlet Analysis")
        self.btn_run_singlet.clicked.connect(self.run_singlet_analysis)
        self.btn_run_singlet.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
        button_layout.addWidget(self.btn_run_singlet)
        
        self.btn_run_doublet = QPushButton("Run Doublet Analysis")
        self.btn_run_doublet.clicked.connect(self.run_doublet_analysis)
        self.btn_run_doublet.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 10px; }")
        button_layout.addWidget(self.btn_run_doublet)
        
        self.control_layout.addLayout(button_layout)
        
        # === Results Display ===
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        self.control_layout.addWidget(results_group)
        
        # Add stretch at bottom to push everything up
        self.control_layout.addStretch()
        
        self.control_scroll.setWidget(self.control_panel)
    
    def setup_plot_panel(self):
        """Setup the right plot panel with matplotlib canvases"""
        self.plot_tabs = QTabWidget()
        
        # Create tabs for different plots
        self.tab_raw = QWidget()
        self.tab_corrected = QWidget()
        self.tab_normalized = QWidget()
        self.tab_fit = QWidget()
        
        # Setup each tab
        self.setup_raw_plot_tab()
        self.setup_corrected_plot_tab()
        self.setup_normalized_plot_tab()
        self.setup_fit_plot_tab()
        
        # Add tabs
        self.plot_tabs.addTab(self.tab_raw, "Raw Data")
        self.plot_tabs.addTab(self.tab_corrected, "Offset Corrected")
        self.plot_tabs.addTab(self.tab_normalized, "Normalized")
        self.plot_tabs.addTab(self.tab_fit, "Fitted Results")
    
    def setup_raw_plot_tab(self):
        """Setup the raw data plot tab"""
        layout = QVBoxLayout()
        self.fig_raw = Figure(figsize=(8, 6))
        self.canvas_raw = FigureCanvas(self.fig_raw)
        self.ax_raw = self.fig_raw.add_subplot(111)
        self.ax_raw.set_xlabel('Frequency (Hz)')
        self.ax_raw.set_ylabel('Power (dBm)')
        self.ax_raw.set_title('Raw Data')
        self.ax_raw.grid(True, alpha=0.3)
        layout.addWidget(self.canvas_raw)
        self.tab_raw.setLayout(layout)
    
    def setup_corrected_plot_tab(self):
        """Setup the offset corrected plot tab"""
        layout = QVBoxLayout()
        self.fig_corrected = Figure(figsize=(8, 6))
        self.canvas_corrected = FigureCanvas(self.fig_corrected)
        self.ax_corrected = self.fig_corrected.add_subplot(111)
        self.ax_corrected.set_xlabel('Frequency (Hz)')
        self.ax_corrected.set_ylabel('T (linear scale)')
        self.ax_corrected.set_title('Offset Corrected Transmission')
        self.ax_corrected.grid(True, alpha=0.3)
        layout.addWidget(self.canvas_corrected)
        self.tab_corrected.setLayout(layout)
    
    def setup_normalized_plot_tab(self):
        """Setup the normalized transmission plot tab"""
        layout = QVBoxLayout()
        self.fig_normalized = Figure(figsize=(8, 6))
        self.canvas_normalized = FigureCanvas(self.fig_normalized)
        self.ax_normalized = self.fig_normalized.add_subplot(111)
        self.ax_normalized.set_xlabel('Frequency (Hz)')
        self.ax_normalized.set_ylabel('T (linear scale)')
        self.ax_normalized.set_title('Normalized Transmission (Savgol Corrected)')
        self.ax_normalized.grid(True, alpha=0.3)
        layout.addWidget(self.canvas_normalized)
        self.tab_normalized.setLayout(layout)
    
    def setup_fit_plot_tab(self):
        """Setup the fitted results plot tab"""
        layout = QVBoxLayout()
        self.fig_fit = Figure(figsize=(8, 6))
        self.canvas_fit = FigureCanvas(self.fig_fit)
        self.ax_fit = self.fig_fit.add_subplot(111)
        self.ax_fit.set_xlabel('Frequency (Hz)')
        self.ax_fit.set_ylabel('T (linear scale)')
        self.ax_fit.set_title('Fitted Resonance')
        self.ax_fit.grid(True, alpha=0.3)
        layout.addWidget(self.canvas_fit)
        self.tab_fit.setLayout(layout)
    
    def load_data_file(self):
        """Load the main data file"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "", "Text Files (*.txt *.csv);;All Files (*)"
        )
        if file_name:
            self.data_file = file_name
            # Display shortened filename
            import os
            short_name = os.path.basename(file_name)
            self.data_file_label.setText(short_name)
            self.results_text.append(f"Data file loaded: {short_name}")
            
            # Clear previous plots
            self.clear_all_plots()
    
    def load_offset_file(self):
        """Load the offset/background file (optional)"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Offset File", "", "Text Files (*.txt *.csv);;All Files (*)"
        )
        if file_name:
            self.offset_file = file_name
            # Display shortened filename
            import os
            short_name = os.path.basename(file_name)
            self.offset_file_label.setText(short_name)
            self.results_text.append(f"Offset file loaded: {short_name}")
    
    def clear_all_plots(self):
        """Clear all plot canvases"""
        for ax in [self.ax_raw, self.ax_corrected, self.ax_normalized, self.ax_fit]:
            ax.clear()
            ax.grid(True, alpha=0.3)
        
        self.ax_raw.set_xlabel('Frequency (Hz)')
        self.ax_raw.set_ylabel('Power (dBm)')
        self.ax_raw.set_title('Raw Data')
        
        self.ax_corrected.set_xlabel('Frequency (Hz)')
        self.ax_corrected.set_ylabel('T (linear scale)')
        self.ax_corrected.set_title('Offset Corrected Transmission')
        
        self.ax_normalized.set_xlabel('Frequency (Hz)')
        self.ax_normalized.set_ylabel('T (linear scale)')
        self.ax_normalized.set_title('Normalized Transmission (Savgol Corrected)')
        
        self.ax_fit.set_xlabel('Frequency (Hz)')
        self.ax_fit.set_ylabel('T (linear scale)')
        self.ax_fit.set_title('Fitted Resonance')
        
        for canvas in [self.canvas_raw, self.canvas_corrected, self.canvas_normalized, self.canvas_fit]:
            canvas.draw()
    
    def get_param_find_peaks(self):
        """Get current peak finding parameters from GUI"""
        return {
            'distance': self.peak_distance.value(),
            'prominence': self.peak_prominence.value(),
            'width': self.peak_width.value(),
            'rel_height': self.peak_rel_height.value()
        }
    
    def get_param_rel_singlet(self):
        """Get current singlet relative parameters from GUI"""
        return {
            'Rel_A': self.singlet_rel_a.value(),
            'Rel_FWHM': self.singlet_rel_fwhm.value(),
            'Rel_Peak': self.singlet_rel_peak.value(),
            'factor_remove_points': 1
        }
    
    def get_param_find_doublet(self):
        """Get current doublet finding parameters from GUI"""
        return {
            'distance': self.doublet_distance.value(),
            'prominence': self.doublet_prominence.value(),
            'width': self.doublet_width.value(),
            'rel_height': self.doublet_rel_height.value()
        }
    
    def get_param_rel_doublet(self):
        """Get current doublet relative parameters from GUI"""
        return {
            'Rel_A1': self.doublet_rel_a1.value(),
            'Rel_A2': self.doublet_rel_a2.value(),
            'Rel_Peak1': self.doublet_rel_peak1.value(),
            'Rel_Peak2': self.doublet_rel_peak2.value(),
            '1st peak': self.doublet_1st_peak.value(),
            '2nd peak': self.doublet_2nd_peak.value()
        }
    
    def run_singlet_analysis(self):
        """Run the complete singlet resonance analysis pipeline"""
        if self.data_file is None:
            self.results_text.append("ERROR: Please load a data file first!")
            return
        
        try:
            self.results_text.clear()
            self.results_text.append("="*50)
            self.results_text.append("Starting Singlet Analysis...")
            self.results_text.append("="*50)
            
            # Initialize analyzer
            self.analyzer = SingleHighQ()
            
            # Get model and parameters
            model = self.model_combo.currentText()
            center_wl = self.wl_spinbox.value()
            
            # Load data
            self.results_text.append(f"\nLoading data with model: {model}")
            self.T_Data = self.analyzer.load_data_singleQ(
                file_name=self.data_file,
                ax=self.ax_raw,
                model=model
            )
            self.canvas_raw.draw()
            
            # Load offset if provided
            if self.offset_file:
                self.results_text.append("Loading offset file...")
                self.T_Offset = self.analyzer.load_data_Offset(ax=self.ax_raw)
                self.canvas_raw.draw()
                
                # Remove offset
                self.results_text.append("Removing offset...")
                self.T_corrected = self.analyzer.Remove_Offset(ax=self.ax_corrected)
                self.canvas_corrected.draw()
            else:
                self.results_text.append("No offset file - using raw data as corrected data")
                self.T_corrected = self.T_Data.copy()
                self.ax_corrected.plot(self.T_corrected['nu_Hz'], 
                                      self.T_corrected['T_linear'],
                                      label='Raw data (no offset correction)')
                self.ax_corrected.legend()
                self.canvas_corrected.draw()
            
            # Get parameters
            param_peaks = self.get_param_find_peaks()
            param_rel = self.get_param_rel_singlet()
            
            self.results_text.append(f"\nPeak finding parameters: {param_peaks}")
            self.results_text.append(f"Relative fitting parameters: {param_rel}")
            
            # Run fitting pipeline
            self.results_text.append("\nRunning fitting pipeline...")
            self.T_normalised = self.analyzer.PL_FitResonance(
                T_corrected=self.T_corrected,
                center_wl_nm=center_wl,
                ax_T=self.ax_normalized,
                ax_T_normalised=self.ax_fit,
                param_rel=param_rel,
                **param_peaks
            )
            
            self.canvas_normalized.draw()
            self.canvas_fit.draw()
            
            self.results_text.append("\n" + "="*50)
            self.results_text.append("Singlet Analysis Complete!")
            self.results_text.append("="*50)
            self.results_text.append("\nCheck the 'Fitted Results' tab for Q-factor and fit details.")
            
        except Exception as e:
            self.results_text.append(f"\nERROR during analysis: {str(e)}")
            import traceback
            self.results_text.append(traceback.format_exc())
    
    def run_doublet_analysis(self):
        """Run doublet resonance analysis on the normalized data"""
        if self.T_normalised is None:
            self.results_text.append("\nERROR: Please run Singlet Analysis first to get normalized data!")
            return
        
        try:
            self.results_text.append("\n" + "="*50)
            self.results_text.append("Starting Doublet Analysis...")
            self.results_text.append("="*50)
            
            # Get parameters
            param_find_doublet = self.get_param_find_doublet()
            param_rel_doublet = self.get_param_rel_doublet()
            center_wl = self.wl_spinbox.value()
            
            self.results_text.append(f"\nDoublet finding parameters: {param_find_doublet}")
            self.results_text.append(f"Doublet relative parameters: {param_rel_doublet}")
            
            # Clear fit axis and replot
            self.ax_fit.clear()
            self.ax_fit.grid(True, alpha=0.3)
            self.ax_fit.set_xlabel('Frequency (Hz)')
            self.ax_fit.set_ylabel('T (linear scale)')
            self.ax_fit.set_title('Doublet Resonance Fit')
            
            # Run doublet fitting
            self.results_text.append("\nFitting doublet resonance...")
            opt_Doublet, pcov_Doublet, f_func = self.analyzer.Q_FitDoubletResonance(
                T_normalised=self.T_normalised,
                center_wl_nm=center_wl,
                param_find_DoubletResonance=param_find_doublet,
                param_rel_Doublet=param_rel_doublet,
                ax_Doublet=self.ax_fit
            )
            
            self.canvas_fit.draw()
            
            self.results_text.append("\n" + "="*50)
            self.results_text.append("Doublet Analysis Complete!")
            self.results_text.append("="*50)
            self.results_text.append("\nCheck the 'Fitted Results' tab for doublet fit details.")
            
        except Exception as e:
            self.results_text.append(f"\nERROR during doublet analysis: {str(e)}")
            import traceback
            self.results_text.append(traceback.format_exc())


def main():
    """Main function to run the GUI application"""
    app = QApplication(sys.argv)
    window = GUI_SingleHighQ()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()