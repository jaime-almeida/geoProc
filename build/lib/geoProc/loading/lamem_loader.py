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
from vtk import vtkXMLStructuredGridReader
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

unit_dict = {
    "velocity":"cm/yr",
    "density":"kg/m^3",
    "stress":"MPa",
    "strain_rate":"1/s",
    "tot_disp":"km",
    "creep":"Pa*s",
    "yield":"MPa",
    "diss":"W/m^3",
    "visc_total":"Pa*s",
    "pressure":"MPa"    
}


# %%
class LaMEMLoader:

    def __init__(self, model_dir, ts=None, load_vars=None, combined_id_names: list = None, verbose=True):
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
        self.dim = 3
        if combined_id_names:
            self.combination_list = combined_id_names

        # TODO: TS AS INT IS NOT WORKING

        # Check how many timesteps to load
        if type(ts) == list:
            load_list = True
            load_single = False
            if verbose:
                print('Loading ts: {}'.format(ts))

        # A single timestep is set
        elif type(ts) == int or type(ts) == float:
            load_list = False
            load_single = True
            if verbose:
                print('Loading ts: {}'.format(ts))

        # No timestep is set
        elif ts is None:
            load_list = False
            load_single = False

        # Get the timestep folder
        # folder_name, time = get_ts_folder(self.model_dir, ts)
        model_folders = get_all_ts_folders(self.model_dir)

        # Store teh model folders:
        self._model_folders = model_folders

        # Define the file extension:
        vtk_ext = 'pvtr'

        # Prepare the dictionary with all outputs:
        self._complete_output = {}
        self._complete_topography = {}
        self.time_stamps = {}
        self.time_Ma = None
        self.current_ts = None

        for folder_name in model_folders:
            # Get current timestep:
            timestep = str(int(folder_name.split('_')[1])).zfill(5)
            time = float(folder_name.split('_')[-1])

            if load_single:
                self.current_ts = ts

                # Make it so only the chosen timestep is loaded
                if int(timestep) != ts:
                    continue

            elif load_list:
                if int(timestep) not in ts:
                    continue

            if verbose:
                print('Now reading timestep {} for model {}'.format(timestep, self.model_dir.split('\\')[-1]))

            # Get the vtk_name from the folder:
            # Warning, this shit is spit-glued.
            # For some ungodly reason it seemed to bug out in the prior iteration)
            file_list = np.sort(os.listdir('{}\\{}'.format(model_dir, folder_name)))
            check_files = [vtk_ext in files for files in file_list]
            vtk_name = file_list[check_files][0]

            # Create the filename to read
            filename = '{}\\{}\\{}'.format(model_dir, folder_name, vtk_name)

            # Start the reader:
            reader = vtkXMLPRectilinearGridReader()

            # Get the information from it:
            reader.SetFileName(filename)

            # print('Compiling data...')
            reader.Update()

            # Get the data
            self._data = reader.GetOutput()

            # Initiate a boundary coordinate
            self.boundary = {}

            # Prepare the output dataframe:
            self.output = None
            self._starting_output = None

            # Get the mesh
            self._get_mesh()

            # Get the variables
            if not load_vars:
                if verbose:
                    print('Getting all variables...')

                self.get_all()
            else:
                coded_vars = ['velocity', 'strain_rate', 'stress', 'temperature']
                
                for var in load_vars:
                    if var in coded_vars:
                        eval('self._get_{}()'.format(var))
                        
                    else:
                        
                        # Check unit for this var:
                        for key in unit_dict.keys():
                            if key in var:
                                # If you find the unit, break the loop
                                unit = unit_dict[key]
                                break
                            else:
                                unit = " "
                        self._get_single_col_var("{} [{}]".format(var, unit), var.split("[")[0].strip())
                        
            # Get the combined ids if needed:
            if combined_id_names:
                self._get_combined_id_values()

            self._complete_output[timestep] = self.output.copy()

            self.time_stamps[timestep] = time

        # Set the current timestep to start the processing, if requested:
        if load_single:
            self.set_current_ts(step=str(self.current_ts).zfill(5))

    def set_current_ts(self, step):
        """
        Function to reset the model output and replace the output object.

        """
        # Reinstanciate the object with a new timestep:
        step = str(step).zfill(5)
        self.output = self._complete_output[step].copy()

        self.current_ts = step
        self.time_Ma = self.time_stamps[step]

    ##################################################
    #              RETRIEVING INFORMATION            #
    ##################################################

    def get_all(self):
        """
        Function to get all existing variables from the current working directory.
        """

        self._get_velocity()
        self._get_temperature()
        self._get_strain_rate()
        self._get_stress()

    def _get_single_col_var(self, pv_name, column_name):
        # Get the phase from the data files:
        new_var = v2n.vtk_to_numpy(self._data.GetPointData().GetArray(pv_name))

        # Save the phase dataframe as 'mat' to keep it working with uw scripts
        new_var = pd.DataFrame(data=new_var, columns=[column_name])

        # Merge with the current output dataframe
        self.output = self.output.merge(new_var, left_index=True, right_index=True)
        
        pass
    
    
    def _get_velocity(self):
        # Get the velocity from the data files:
        velocity = v2n.vtk_to_numpy(self._data.GetPointData().GetArray('velocity [cm/yr]'))

        # in 3D:
        velocity = pd.DataFrame(data=velocity, columns=['vx', 'vy', 'vz'])

        # Merge with the current output dataframe
        self.output = self.output.merge(velocity, left_index=True, right_index=True)

    
    
    def _get_mesh(self):
        # Prepare to receive the mesh:
        x = np.zeros(self._data.GetNumberOfPoints())
        y = np.zeros(self._data.GetNumberOfPoints())
        z = np.zeros(self._data.GetNumberOfPoints())

        # Build the mesh:
        for i in range(self._data.GetNumberOfPoints()):
            x[i], y[i], z[i] = self._data.GetPoint(i)

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


    def _get_strain_rate(self):
        # Order is: xx xy xz yx yy yz zx zy zz
        # Get the phase from the data files:
        strain_rate = v2n.vtk_to_numpy(self._data.GetPointData().GetArray('strain_rate [1/s]'))
        strain_rate_invariant = v2n.vtk_to_numpy(self._data.GetPointData().GetArray('j2_strain_rate [1/s]'))

        # Save the phase dataframe as 'mat' to keep it working with uw scripts
        strain_rate = pd.DataFrame(data=strain_rate,
                                   columns=['e_xx', 'e_xy', 'e_xz', 'e_yx', 'e_yy', 'e_yz', 'e_zx', 'e_zy', 'e_zz'])

        strain_rate_invariant = pd.DataFrame(data=strain_rate_invariant,
                                             columns=['e_II'])

        # Merge with the current output dataframe
        self.output = self.output.merge(strain_rate, left_index=True, right_index=True)
        self.output = self.output.merge(strain_rate_invariant, left_index=True, right_index=True)

    def _get_stress(self):
        # Order is: xx xy xz yx yy yz zx zy zz
        # Get the phase from the data files:
        strain_rate = v2n.vtk_to_numpy(self._data.GetPointData().GetArray('dev_stress [MPa]'))
        strain_rate_invariant = v2n.vtk_to_numpy(self._data.GetPointData().GetArray('j2_dev_stress [MPa]'))

        # Save the phase dataframe as 'mat' to keep it working with uw scripts
        strain_rate = pd.DataFrame(data=strain_rate,
                                   columns=['s_xx', 's_xy', 's_xz', 's_yx', 's_yy', 's_yz', 's_zx', 's_zy', 's_zz'])

        strain_rate_invariant = pd.DataFrame(data=strain_rate_invariant,
                                             columns=['s_II'])

        # Merge with the current output dataframe
        self.output = self.output.merge(strain_rate, left_index=True, right_index=True)
        self.output = self.output.merge(strain_rate_invariant, left_index=True, right_index=True)

    def _get_temperature(self):
        # Get the phase from the data files:
        temperature = v2n.vtk_to_numpy(self._data.GetPointData().GetArray('temperature [C]'))

        # Save the phase dataframe as 'mat' to keep it working with uw scripts
        temperature = pd.DataFrame(data=temperature, columns=['temp_C'])
        temperature['temp_K'] = temperature.temp_C - 273.15

        # Merge with the current output dataframe
        self.output = self.output.merge(temperature, left_index=True, right_index=True)

    @property
    def complete_output(self):
        return self._complete_output

    ##################################################
    #                 GET THE TOPOGRAPHY             #
    ##################################################
    # TODO: Solve this topography issue
    def get_topography(self):

        # check in model folders which is the correct one:
        correct_folder = [folder for folder in self._model_folders if str(self.current_ts) in folder][0]

        # Get the vtk_name from the folder:
        file_list = os.listdir('{}\\{}'.format(self.model_dir, correct_folder))
        vtk_name = [file for file in file_list if 'pvts' in file][0]

        # Create the filename to read
        filename = '{}\\{}\\{}'.format(self.model_dir, correct_folder, vtk_name)

        # Start the reader:
        reader = vtkXMLStructuredGridReader()

        # Get the information from it:
        reader.SetFileName(filename)

        # print('Compiling data...')
        reader.Update()

        # Get the data
        topo_data = reader.GetOutput()

        # =================== GET THE MESH DATA FOR THE SURFACE ===================

        # Prepare to receive the mesh:
        x = np.zeros(topo_data.GetNumberOfPoints())
        y = np.zeros(topo_data.GetNumberOfPoints())
        z = np.zeros(topo_data.GetNumberOfPoints())

        # Build the mesh:
        for i in range(topo_data.GetNumberOfPoints()):
            x[i], y[i], z[i] = topo_data.GetPoint(i)

        # Create the mesh:
        mesh_info = np.column_stack([x, y, z])

        # Build the 3D array in the output dataframe:
        self.topography = self.topography = pd.DataFrame(data=mesh_info, columns=['x', 'y', 'z'], dtype='float')

        # ======================== GET THE SURFACE DATA ===================
        # Get the phase from the data files:
        topography = v2n.vtk_to_numpy(self._topo_data.GetPointData().GetArray('topography [km]'))

        # Save the phase dataframe as 'mat' to keep it working with uw scripts
        topography = pd.DataFrame(data=topography, columns=['topography'])

        # Merge with the current output dataframe
        self.topography = self.topography.merge(topography, left_index=True, right_index=True)

    ##################################################
    #              GET THE COMBINED NAMES            #
    ##################################################

    def _get_combined_id_values(self):
        # For each item on the list, get the value and append to the output frame:
        for combination_name in self.combination_list:
            data = v2n.vtk_to_numpy(self._data.GetPointData().GetArray('{} [ ]'.format(combination_name)))

            # Save this in the list:
            df = pd.DataFrame(data=data, columns=[combination_name])

            # Merge with the current output dataframe
            self.output = self.output.merge(df, left_index=True, right_index=True)


if __name__ == '__main__':
    test = LaMEMLoader(model_dir=r'F:\GEMMA_REPO\make_videos\model_results\long_tfs_plateau',
                       ts=[0],
                       load_vars=['phase','velocity', 'strain_rate', 'visc_total', 'density', "rel_pl_rate"])

    