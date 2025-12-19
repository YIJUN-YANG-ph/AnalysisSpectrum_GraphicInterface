#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:44:26 2024

@author: yangyijun
"""
'''
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
import pyqtgraph as pg
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('PyQtGraph Plot Example')
        self.setGeometry(100, 100, 800, 600)

        # Create a central widget and set a vertical layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create a PyQtGraph plot widget
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        # Plot some data
        x = np.linspace(0, 10, 1000)
        y = np.sin(x)
        self.plot_widget.plot(x, y)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
'''
import sys
import os
import ast

import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib
from scipy.optimize import curve_fit

import pandas as pd
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                             QPushButton, QFileDialog, QHBoxLayout, QGridLayout,
                             QLineEdit, QLabel)
from PyQt5.QtGui import QFont

import pyqtgraph as pg
import numpy as np
from scipy.interpolate import interp1d
from components_AnalysisSpectrum import (nu2wl,wl2nu,
                                         dB2linear, linear2dB, 
                                         load_data,
                                         find_peaks_for_FSR,
                                         analysis_main)
C = 299792458 #m / s

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.T={}
        
        self.setWindowTitle('CSV Plotter with PyQtGraph')
        self.setGeometry(100, 100, 600, 1200)

        # Create a central widget and set a vertical layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create a PyQtGraph plot widget
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)
        # Set background to white
        self.plot_widget.setBackground('w')
        #inialize the legend
        self.legend = self.plot_widget.addLegend(labelTextSize='10pt')
        self.set_legend_style()
        
        # Create a second pyQtGraph plot widget, for presenting some simple operation on the spectrum.
        self.plot_widget_operation = pg.PlotWidget()
        layout.addWidget(self.plot_widget_operation)
        # Set background to white
        self.plot_widget_operation.setBackground('w')
       

        # Create horizontal layout for the buttons
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)
        # Creat horizontal layout for analysis buttons
        analysis_button_layout = QHBoxLayout()
        layout.addLayout(analysis_button_layout)

# =============================================================================
#         # Create a button for loading CSV
#         by default, the click.connect give a first parameter value False..
#         Therefore, the lambda function is created to avoid given this False by default.
# =============================================================================
        self.button = QPushButton('Import CSV')
        self.button.clicked.connect(lambda: self.load_csv())
        layout.addWidget(self.button)
        
# =============================================================================
#         # Parameter Input Fields
# =============================================================================
        param_layout = QGridLayout()
        self.param_row_counter = 0  # Initialize row counter
        self.param_fields = {}
        # Adding parameter fields dynamically based on initial values
        self.add_param_field(param_layout, "Param_RingResonator", {'diameter': 200, 'width': 1100, 'range_wl': [1500, 1640]})
        self.add_param_field(param_layout, "Param_find_peaks", {'distance': 1600, 'prominence': 0.001, 'width': 4,'rel_height': 1})
        self.add_param_field(param_layout, "Param_Savgol_fitting", {'window_length': 101, 'points_to_remove': 151, 'polyorder': 2})
        self.add_param_field(param_layout, "Param_peaks_fitting", {'peak_range_nm': 1, 'A_margin': 0.02})
        self.add_param_field(param_layout, "Param_FSR_fitting", {'fitting_order': 2, 'nb_sigma': 3})
        self.add_param_field(param_layout, "Param_loss_calcul", {'wl_critical': 1535})
        layout.addLayout(param_layout)

        
        # Create the "Add Offset" button (smaller size)
        self.button_offset = QPushButton('Add Offset')
        self.button_offset.clicked.connect(self.load_offset)
        button_layout.addWidget(self.button_offset)

        # Create the "Operator" button (smaller size)
        self.button_operator = QPushButton('Operator')
        self.button_operator.clicked.connect(lambda: self.calculate_difference_yijun())
        button_layout.addWidget(self.button_operator)
        
        
        
        # Buttons for different plots for analysis
        self.plot_buttons = {
            "Show Figure 1": QPushButton("Analyze"),
            "Show Figure 4": QPushButton("arrange wavelength"),
            "Show Figure 2": QPushButton("Close all"),
            "Show Figure 3": QPushButton("Clear all"),
        }
        for name, button in self.plot_buttons.items():
            button.clicked.connect(self.show_plot)
            analysis_button_layout.addWidget(button)

        # add a button to save the current parameters
        self.button_save_params = QPushButton('Save Parameters')
        analysis_button_layout.addWidget(self.button_save_params)
        # connect the button to a function
        self.button_save_params.clicked.connect(lambda: self.save_current_params())
        # add a button to load the parameters from a text file
        self.button_load_params = QPushButton('Load Parameters')
        analysis_button_layout.addWidget(self.button_load_params)
        # connect the button to a function
        self.button_load_params.clicked.connect(lambda: self.load_params_from_file())
        
        # Load default CSV file on startup
        self.default_csv_path = r'test_W1100_R100um_G400_AfterAnnealing.csv'
        self.load_csv(file_name = self.default_csv_path)
    
    
# =============================================================================
#     The actions of each button
# =============================================================================
    def show_plot(self):
        
        sender = self.sender().text()
        print(f"Generating plot for: {sender}")
        # Implement logic to generate the requested plot based on sender and updated parameters
        
        # Example logic placeholder
        if sender == "Analyze":
            '''
            find resonaces
            '''
            self.retrieve_params()
            self.Fig_FSR,self.Fig_peak_fitting,self.Fig_resonance,self.fig_loss = analysis_main(self.T,
                          self.Param_RingResonator,
                          self.Param_find_peaks,
                          self.Param_Savgol_fitting,
                          self.Param_peaks_fitting,
                          self.Param_FSR_fitting,
                          self.Param_loss_calcul,
                          )   
        elif sender == "Close all":
            # if hasattr(self, 'Fig_FSR') is not None:
            plt.close('all')
        elif sender == "Clear all":
            print('All the data has been deleted, yeah!!!!!!!!!\n')
            self.T = {}
            self.measurement_data = {}
            # Clear any existing plot 
            self.plot_widget_operation.clear()
            self.plot_widget.clear()
            # self.calculate_difference_yijun()
        elif sender == "arrange wavelength":
            self.arrange_wl_range()
    
    
    def add_param_field(self, layout, param_name, default_values):
        # Get the current row count to place new widgets below the existing ones
        # current_row = layout.rowCount()
        
        # Add fields for each parameter in the given dictionary
        for row, (key, value) in enumerate(default_values.items()):
            label = QLabel(f"{param_name} - {key}:")
            field = QLineEdit(str(value))
            layout.addWidget(label, self.param_row_counter, 0)
            layout.addWidget(field, self.param_row_counter, 1)
            layout.addWidget(QLabel('testttt'), row, 2)

            self.param_fields[f"{param_name}_{key}"] = field
            
            # Move to the next row for the next set of widgets
            self.param_row_counter += 1  #  row counter += 1
            
            
    def retrieve_params(self):
        """ 
        To retrieve the input value from a QLineEdit field, you can use the text() method,
        which returns the text in the field as a string. To convert it to a different data type, like int or float, 
        you can wrap it in the desired type conversion.
        
        Keyword Args:
           * **?** : .
        """
        params = {}
        for name, field in self.param_fields.items():
            try:
                # Gets the value as a string
                params[name] = float(field.text())  # Convert to float; change to int() if appropriate
            except ValueError:
                try:
                    # Convert the string to a list using ast.literal_eval, then to a numpy array
                    params[name] = np.array(ast.literal_eval(field.text()))
                except (ValueError, SyntaxError):
                    # Handle errors if the input format is incorrect
                    params[name] = None  # Assign None or a default value if the conversion fails
        
        self.Param_RingResonator = {'diameter': params['Param_RingResonator_diameter'], 
                               'width': params['Param_RingResonator_width'],
                               'range_wl': params['Param_RingResonator_range_wl']}
        self.Param_find_peaks = {'distance': params['Param_find_peaks_distance'],
                            'prominence': params['Param_find_peaks_prominence'],
                            'width': params['Param_find_peaks_width'],
                            'rel_height': params['Param_find_peaks_rel_height']}
        self.Param_Savgol_fitting = {'window_length':params['Param_Savgol_fitting_window_length'],
                        'points_to_remove':params['Param_Savgol_fitting_points_to_remove'],
                        'polyorder':params['Param_Savgol_fitting_polyorder']}
        self.Param_peaks_fitting = {'peak_range_nm':params['Param_peaks_fitting_peak_range_nm'],
                               'A_margin':params['Param_peaks_fitting_A_margin']}
        self.Param_FSR_fitting = {'fitting_order':params['Param_FSR_fitting_fitting_order'],
                             'nb_sigma':params['Param_FSR_fitting_nb_sigma'],}
        self.Param_loss_calcul = {'wl_critical':params['Param_loss_calcul_wl_critical']}
        # Diamter = self.Param_RingResonator['diameter']
        # self.RoundTrip = np.pi*Diamter*1e-6
    
        return params
                
                
                
                
    def set_legend_style(self):
        """Set the legend style, including font size. only for pyqrgraph """
        # Set the font for the legend
        legend_font = QFont()
        legend_font.setPointSize(98)  # Set font size to your preferred value, no use right now
        self.legend.setFont(legend_font)

        # Set offset if you want to change legend position
        self.legend.setOffset((10, 165))  # Adjusts position of the legend box on the plot

    def load_csv(self,file_name = None):
        # If no file is specified, open a dialog to select a CSV
        if file_name is None:
            # enable the file dialog to select multiple files
            options = QFileDialog.Options()
            file_names, _ = QFileDialog.getOpenFileNames(self, "Open CSV File", "", "All Files (*);;CSV Files (*.csv)", options=options)
        else:
            file_names = [file_name]
        T_s = []
        # if file_name:
        for file_name in file_names:
            file_base_name = os.path.basename(file_name)
            '''
            # Check if 'wavelength' and 'transmission' columns are present
            if 'wavelength' in data.columns and 'transmission' in data.columns:
                wavelength = data['wavelength'].values
                transmission = data['transmission'].values
            elif 'L' in data.columns and '1' in data.columns:
                wavelength = data['L'].values
                transmission = data['1'].values
                
            # Plot the data with 'tab:blue' color for the curve
            self.plot_widget.plot(wavelength, transmission, pen=pg.mkPen(color='black',width=2))
            self.measurement_data = pd.DataFrame(data = {'wavelength':wavelength,'transmission':transmission})    
            '''
            try:
                T = load_data(file_name,range_wl=None,wl_name='wavelength',data_name='transmission')
            except ValueError:
                # T = load_data(FileName,range_wl=[1500,1609],wl_name='L',data_name='1')
                print('Loading data fails!\n')
            T_s.append(T)   

        # Clear any existing plot
        self.plot_widget.clear()
        # plot all dataframe T in T_s on the same plot
        # Loop through each DataFrame in T_s and plot it, the color should not be the same, the label size should be the same
        for i, T in enumerate(T_s):
            # Generate a unique color for each plot
            color = pg.intColor(i, hues=len(T_s))
            # Plot the data with the generated color
            file_base_name = os.path.basename(file_names[i])
            self.plot_widget.plot(T['wavelength_nm'], T['T_dB'], pen=pg.mkPen(color=color, width=2), name=file_base_name,
                                  )
        
        # # Plot the data with 'tab:blue' color for the curve
        # self.plot_widget.plot(T['wavelength_nm'],T['T_dB'], pen=pg.mkPen(color=(25, 25, 112),width=2),name=file_base_name,
        #                       symbolsize = 50)
        self.measurement_data = T
        self.measurement_data_s = T_s # a list of dataframes that contains all the datas
    def load_offset(self):
            """Load offset CSV file and plot it in grey."""
            
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "All Files (*);;CSV Files (*.csv)", options=options)
    
            if file_name:
                try:
                    T = load_data(file_name,range_wl=None,wl_name='wavelength',data_name='transmission')
                except ValueError:
                    # T = load_data(FileName,range_wl=[1500,1609],wl_name='L',data_name='1')
                    print('Loading data fails!\n')    
                    # Plot the offset in grey
                self.plot_widget.plot(T['wavelength_nm'],T['T_dB'], pen=pg.mkPen(color='gray', width=2))
                self.offset_data = T
    def save_current_params(self):
        """Save the current parameters to a text file."""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Parameters", "", "Text Files (*.txt);;All Files (*)", options=options)
        
        if file_name:
            try:
                with open(file_name, 'w') as f:
                    for name, field in self.param_fields.items():
                        f.write(f"{name}: {field.text()}\n")
                print(f"Parameters saved to {file_name}")
            except Exception as e:
                print(f"Error saving parameters: {e}")
    def load_params_from_file(self):
        """Load parameters from a text file and update the input fields."""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Parameters", "", "Text Files (*.txt);;All Files (*)", options=options)
        
        if file_name:
            try:
                with open(file_name, 'r') as f:
                    for line in f:
                        name, value = line.strip().split(': ', 1)
                        if name in self.param_fields:
                            self.param_fields[name].setText(value)
                print(f"Parameters loaded from {file_name}")
            except Exception as e:
                print(f"Error loading parameters: {e}")

# =============================================================================
#     new functions defined by Yijun YANG
# =============================================================================
    def calculate_difference_yijun(self):
        """ Calculate the difference between measurement and offset, plot in a new window. If no offset, just plot the measurements.
            Comments by Yijun
            Args:
               * **nu_res_array** (): 
            Outputs:
               * **?** (?): ?
        """
        
        params = self.retrieve_params()
        range_wl = params['Param_RingResonator_range_wl'].tolist()
        # range_wl
        if self.measurement_data_s is not None:
            result_offsetRM_s = [] # mesurement - offset, list
            for measurement in self.measurement_data_s:
                # Check if offset_data exists
                if hasattr(self, 'offset_data') and self.offset_data is not None:
                    # Extract the wavelength and transmission data
                    wavelength_meas = measurement['wavelength_nm'].values
                    transmission_meas = measurement['T_linear'].values
        
                    wavelength_offset = self.offset_data['wavelength_nm'].values
                    transmission_offset = self.offset_data['T_linear'].values
        
                    # Interpolate the offset data to match the measurement wavelength if needed
                    f = interp1d(wavelength_offset, transmission_offset, bounds_error=False, fill_value="extrapolate")
                    interpolated_offset = f(wavelength_meas)
        
                    # Calculate the difference (measurement - offset) in dB
                    result = transmission_meas / interpolated_offset
                else:
                    # If no offset data, use T_linear as result
                    result = measurement['T_linear'].values
                result_dB = linear2dB(result)
                # result_offsetRM_s.append(result)# append the linear result

                # give the prenoramlised transmission back to measurement_data
                # Have a problem!!!!!!!!!!!!!!!!!!!!!!!!!!!

                # self.measurement_data['T_dB']=result_dB
                # self.measurement_data['T_linear']=result
                measurement_data = measurement.copy()
                measurement_data['T_dB']=result_dB
                measurement_data['T_linear']=result

                # Apply the range filter if specified
                if range_wl is not None:
                    mask = (measurement_data['wavelength_nm'].values >= range_wl[0]) & (measurement_data['wavelength_nm'].values <= range_wl[1])
                    # Create a new DataFrame with the filtered data and reset index
                    self.T = measurement_data[mask].copy().reset_index(drop=True)
                    result = result.copy()[mask]
                else:
                    # No range filter, T contains all measurement data
                    # Create a new DataFrame with the filtered data and reset index
                    self.T = measurement_data.copy().reset_index(drop=True)


                result_offsetRM_s.append(result)# append the linear result 
            # Clear any existing plot and plot the result within the selected range
            self.plot_widget_operation.clear()
            #pg.PlotWidget() can not handle pd.dataframe, so it has to be convert to np.array
            for i, result in enumerate(result_offsetRM_s):
                # Generate a unique color for each plot
                color = pg.intColor(i, hues=len(result_offsetRM_s))
                # Plot the data with the generated color
                self.plot_widget_operation.plot(self.T['wavelength_nm'].to_numpy(), result, pen=pg.mkPen(color=color, width=2))
            # self.plot_widget_operation.plot(self.T['wavelength_nm'].to_numpy(), self.T['T_linear'].to_numpy(), pen=pg.mkPen(color=(160,32,240), width=2))
    def arrange_wl_range(self):
        params = self.retrieve_params()
        range_wl = params['Param_RingResonator_range_wl'].tolist()
        try:
            # Apply the range filter if specified
            if range_wl is not None:
                print(f'wavelength range: {range_wl[0]} to {range_wl[1]} nm\n')
                mask = (self.measurement_data['wavelength_nm'].values >= range_wl[0]) & (self.measurement_data['wavelength_nm'].values <= range_wl[1])
                # Create a new DataFrame with the filtered data and reset index
                self.T = self.measurement_data[mask].copy().reset_index(drop=True)
                # Clear any existing plot and plot the result within the selected range
                self.plot_widget_operation.clear()
                #pg.PlotWidget() can not handle pd.dataframe, so it has to be convert to np.array
                self.plot_widget_operation.plot(self.T['wavelength_nm'].to_numpy(), self.T['T_linear'].to_numpy(), pen=pg.mkPen(color=(160, 32, 240), width=2))
            else:
                self.T = self.measurement_data.copy().reset_index(drop=True)
                self.plot_widget_operation.plot(self.T['wavelength_nm'].to_numpy(), self.T['T_linear'].to_numpy(), pen=pg.mkPen(color=(160, 32, 240), width=2))

        except (ValueError, SyntaxError):
            # Handle errors if the input format is incorrect
            self.T = self.T  # Assign None or a default value if the conversion fails
            print('arrange wavelength range fails!\n')
            
            
                
            
                
            
if __name__ == "__main__":
    # Check if a QApplication instance already exists, if not create one
    # This is to avoid creating multiple QApplication instances, elsewise kernel wi
    matplotlib.use('Qt5Agg')# use the Qt5Agg backend, important

    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance() 
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

