# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:34:51 2019

@author: jaime
#"""
import pandas as pd
import re as re
from sys import platform
import numpy as np
from vtk import vtkXMLPRectilinearGridReader
from vtk.util import numpy_support as v2n
import os


class UserChoice(Exception):
    def __init__(self, message):
        self.message = message


# Print iterations progress
def get_model_name(model_dir):
    if 'win' in platform:
        if model_dir[-1] == r'\\':
            model_dir -= r'\\'

        return re.split(r'\\', model_dir)[-1]
    else:
        if model_dir[-1] == '/':
            model_dir -= '/'

        return re.split('/', model_dir)[-1]


def ts_writer(ts_in):
    # Making the timestep text:
    return str(ts_in).zfill(5)


def get_ts_folder(model_dir, ts):
    # Check which folders are within the model folder:
    for line in os.listdir(model_dir):
        # If it is a Timestep* folder
        if not os.path.isfile(line) and line[0] == 'T':
            columns = line.split('_')
            current_ts = int(columns[1])
            current_time = float(columns[-1])

            if current_ts == int(ts):
                # If it is the one we want:
                output = line
                break
    else:
        raise (Exception, 'Timestep not found.')
    return output, current_time


def get_all_ts_folders(model_dir):
    # Check which folders are within the model folder:
    needed_folders = []
    for line in os.listdir(model_dir):
        # If it is a Timestep* folder
        if not os.path.isfile(line) and line[0] == 'T':
            columns = line.split('_')
            current_ts = int(columns[1])
            current_time = float(columns[-1])
            needed_folders.append(line)
    return needed_folders


# %%
class LaMEMLoader:

    def __init__(self, model_dir, ts=0, model_zone='internal'):
        """
        Loading function to generate the input for LaMEM model processing.
        Input arguments:
            - model_dir: path to the model folder
            - vtk_name: name of the lamem output
            - ts: timestep to process
            - model_zone: type of model output to open (internal or surface)
        """

        # Verify if the path is correct:
        if not os.path.isdir(model_dir):
            raise FileNotFoundError('No such model exists.')

        # SAve the input variables:
        self.model_dir = model_dir
        self.current_ts = ts
        self.dim = 3

        # Check how many timesteps to load
        if ts:
            load_single = True
        else:
            load_single = False

        # Get the timestep folder
        # folder_name, time = get_ts_folder(self.model_dir, ts)
        model_folders = get_all_ts_folders(self.model_dir)

        # Define the file extension:
        if model_zone == 'internal':
            vtk_ext = 'pvtr'
        elif model_zone == 'surface':
            vtk_ext = 'pvts'
        else:
            raise (TypeError, 'Non-existant LaMEM extension.')

        # Prepare the dictionary with all outputs:
        self._complete_output = {}
        self.time_stamps = {}
        self.time_Ma = None

        for folder_name in model_folders:
            # Get current timestep:
            timestep = str(int(folder_name.split('_')[1])).zfill(5)
            time = float(folder_name.split('_')[-1])

            if load_single:
                self.current_ts = ts
                # Make it so only the chosen timestep is loaded
                if int(timestep) != ts:
                    continue

            print('Now reading timestep {} for model {}'.format(timestep, self.model_dir.split('\\')[-1]))

            # Get the vtk_name from the folder:
            file_list = os.listdir('{}\\{}'.format(model_dir, folder_name))
            check_files = [vtk_ext in files for files in file_list]
            vtk_name = file_list[check_files == 1]

            # Create the filename to read
            filename = '{}\\{}\\{}'.format(model_dir, folder_name, vtk_name)

            # Start the reader:
            reader = vtkXMLPRectilinearGridReader()

            # Get the information from it:
            reader.SetFileName(filename)

            # print('Compiling data...')
            reader.Update()

            self.__data = reader.GetOutput()

            # Initiate a boundary coordinate
            self.boundary = {}

            # Prepare the output dataframe:
            self.output = None
            self._starting_output = None

            # Get the mesh
            self._get_mesh()

            # Get the variables
            self.get_all()

            self.complete_output[timestep] = self.output.copy()
            self.time_stamps[timestep] = time

        # Set the current timestep to start the processing, if requested:
        self.set_current_ts(step=str(self.current_ts).zfill(5))

    def set_current_ts(self, step):
        """
        Function to reset the model output and replace the output object.

        """
        # Reinstanciate the object with a new timestep:
        step = str(step).zfill(5)
        self.output = self.complete_output[step].copy()
        self.current_ts = step
        self.time_Ma = self.time_stamps[step]

    ##################################################
    #              RETRIEVING INFORMATION            #
    ##################################################

    def get_all(self):
        """
        Function to get all existing variables from the current working directory.
        """
        print('Getting all variables...')
        self._get_velocity()
        self._get_phase()
        self._get_viscosity()
        self._get_pressure()
        self._get_temperature()
        self._get_strain_rate()

    def _get_velocity(self):
        # Get the velocity from the data files:
        velocity = v2n.vtk_to_numpy(self.__data.GetPointData().GetArray('velocity [cm/yr]'))

        # in 3D:
        velocity = pd.DataFrame(data=velocity, columns=['vx', 'vy', 'vz'])

        # Merge with the current output dataframe
        self.output = self.output.merge(velocity, left_index=True, right_index=True)

    def _get_mesh(self):
        # Prepare to receive the mesh:
        x = np.zeros(self.__data.GetNumberOfPoints())
        y = np.zeros(self.__data.GetNumberOfPoints())
        z = np.zeros(self.__data.GetNumberOfPoints())

        # Build the mesh:
        for i in range(self.__data.GetNumberOfPoints()):
            x[i], y[i], z[i] = self.__data.GetPoint(i)

        # Create the mesh:
        mesh_info = np.column_stack([x, y, z])

        # Build the 3D array in the output dataframe:
        self.output = self.output = pd.DataFrame(data=mesh_info, columns=['x', 'y', 'z'], dtype='float')

        # Save the model dimensions / boundaries:
        axes = self.output.columns.values
        max_dim = self.output.max().values
        min_dim = self.output.min().values

        for axis, min_val, max_val in zip(axes, min_dim, max_dim):
            self.boundary[axis] = [min_val, max_val]

    def _get_phase(self):
        # Get the phase from the data files:
        phase = v2n.vtk_to_numpy(self.__data.GetPointData().GetArray('phase [ ]'))

        # Save the phase dataframe as 'mat' to keep it working with uw scripts
        phase = pd.DataFrame(data=phase, columns=['mat'])

        # Merge with the current output dataframe
        self.output = self.output.merge(phase, left_index=True, right_index=True)

    def _get_viscosity(self):
        # Get the phase from the data files:
        eta = v2n.vtk_to_numpy(self.__data.GetPointData().GetArray('visc_total [Pa*s]'))

        # Save the phase dataframe as 'mat' to keep it working with uw scripts
        eta = pd.DataFrame(data=eta, columns=['eta'])

        # Merge with the current output dataframe
        self.output = self.output.merge(eta, left_index=True, right_index=True)

    def _get_pressure(self):
        # Get the phase from the data files:
        pressure = v2n.vtk_to_numpy(self.__data.GetPointData().GetArray('pressure [MPa]'))

        # Save the phase dataframe as 'mat' to keep it working with uw scripts
        pressure = pd.DataFrame(data=pressure, columns=['pressure_MPa'])

        # Merge with the current output dataframe
        self.output = self.output.merge(pressure, left_index=True, right_index=True)

    def _get_strain_rate(self):
        # Order is: xx xy xz yx yy yz zx zy zz
        # Get the phase from the data files:
        strain_rate = v2n.vtk_to_numpy(self.__data.GetPointData().GetArray('strain_rate [1/s]'))

        # Save the phase dataframe as 'mat' to keep it working with uw scripts
        strain_rate = pd.DataFrame(data=strain_rate,
                                   columns=['e_xx', 'e_xy', 'e_xz', 'e_yx', 'e_yy', 'e_yz', 'e_zx', 'e_zy', 'e_zz'])

        # Merge with the current output dataframe
        self.output = self.output.merge(strain_rate, left_index=True, right_index=True)

    def _get_temperature(self):
        # Get the phase from the data files:
        temperature = v2n.vtk_to_numpy(self.__data.GetPointData().GetArray('temperature [C]'))

        # Save the phase dataframe as 'mat' to keep it working with uw scripts
        temperature = pd.DataFrame(data=temperature, columns=['temp_C'])
        temperature['temp_K'] = temperature.temp_C - 273.15

        # Merge with the current output dataframe
        self.output = self.output.merge(temperature, left_index=True, right_index=True)

    @property
    def complete_output(self):
        return self._complete_output


#
if __name__ == '__main__':
    test = LaMEMLoader(model_dir='Z:\\PlateauCollision3D_LM\\model_results\\plateau_size\\_L_D70_O70\\',
                       ts=400
                       )
    test.output
