from geoProc.loading.uw_loader import *
from geoProc.loading.lamem_loader import LaMEMLoader

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def get_closest(df, value):
    # Differences between them
    delta = np.abs(df - value)

    # get the index for the closest value:
    depth_id = delta.abs().sort_values().index[0]

    return df.iloc[depth_id]


def find_trailing_edges(coordinates, velocities):
    """
        Get the trailing edges by finding where the ridge is. First moving thing is the trailing edge
    """
    if type(coordinates) == pd.Series or type(velocities) == pd.Series:
        velocities = velocities.to_numpy()
        coordinates = coordinates.to_numpy()

    vx = velocities
    x = coordinates

    # Calculate the velocity divergence
    dvx = np.gradient(vx)
    dx = np.gradient(x)
    div_v = -dvx / dx

    peaks = np.zeros(1, )
    prominence = 0.1
    n_tries = 0
    while len(peaks) <= 3 >= len(np.unique(peaks)):
        prominence += 0.05

        # Find the peaks (5% of max distance apart)
        peaks, _ = find_peaks(np.abs(div_v) / np.max(np.abs(div_v)),
                              prominence=(prominence, None),
                              distance=int(len(div_v) / 20))
        # print(x[peaks])
        if prominence == 1 or n_tries > 20:
            raise Exception('Peak detection issue.')

        n_tries += 1

    else:
        print('Sufficient peaks found @ prominence {:.2f} with {} tries.'.format(prominence, n_tries))

    # Trailing edges, the second from each side?:
    edges = np.array([x[peaks[0]], x[peaks[-1]]])

    # if len(peaks) == 3:
    #     # Trench should be the central
    #     trench = x[peaks][1]
    # else:
    #     trench = float(x[peaks[1:-1]][np.abs(div_v[peaks][1:-1]) == np.abs(div_v[peaks][1:-1]).max()])

    # If there's more than one peak between the edges
    if len(peaks[1:-1]) >= 2:

        # If the second peak is far from the edges
        # if peaks[1] > int(len(x) / 5):
        first_trench_peak = 2

        # Trench is somewhere between the two central peaks:
        trench = peaks[first_trench_peak] + int((peaks[first_trench_peak + 1] - peaks[first_trench_peak]) / 2)
    elif len(peaks[1:-1]) < 2:
        # if it's just the one
        trench = peaks[1]

    else:
        raise Exception('Stupid amount of peaks, please check')

    return edges, x[trench]


def get_time(mdir, ts):
    data = h5.File(mdir + 'timeInfo.' + ts + '.h5', 'r')

    time_out = data['currentTime'][0]

    return time_out


class ModelProcessing:

    def __init__(self, model_dir=None, **kwargs):

        # Save it:
        self.performed_slices = []
        self.model_dir = model_dir

        # Detect what loader to use - simply check if there's an h5 file in the folder:
        if sum(['h5' in file_name for file_name in os.listdir(model_dir)]) != 0:
            self.loader = 'uw'
        else:
            self.loader = 'lamem'

        if self.loader == 'uw':
            temp = UwLoader(model_dir=model_dir, **kwargs)
            self.current_step = temp.current_step
            self.output = temp.output
            self.time_Ma = temp.time_Ma
            self.dim = temp.dim
            self._starting_output = temp.starting_output.copy()
            self.scf = temp.scf
            self.model_name = temp.model_name

        elif self.loader == 'lamem':
            temp = LaMEMLoader(model_dir=model_dir, **kwargs)
            self.output = temp.output
            # self.time_Ma = temp.time_Ma
            self.dim = temp.dim
            self._complete_output = temp.complete_output.copy()
            self.time_stamps = temp.time_stamps

            if temp.current_ts:
                # if it came from a load_single stage
                self.current_step = temp.current_ts

        # Get rid of this huge thing
        del temp

    def set_current_ts(self, step):
        """
        Function to reset the model output and replace the output object.

        """
        if self.loader == 'lamem':
            # Reinstanciate the object with a new timestep:
            step = str(step).zfill(5)
            self.output = self._complete_output[step].copy()
            self.current_step = step
            self.time_Ma = self.time_stamps[step]

        else:
            # Get the current time and descale it
            self.time_Ma = np.round(get_time(self.model_dir, self.current_step) * self.scf / (365 * 24 * 3600) / 1e6, 3)

    def interpolate_window(self, variable, hdir='x', vdir='y', n_elements=int(1e3)):
        """
        Reinterpolate the dataframes into 2D numpy arrays for easier plotting

        returns:
            hmesh, vmesh, var

        """
        # if variable == 'all':
        # variable = dict.fromkeys(self.output.keys())

        # if type(variable) == list:
        # variable = dict.fromkeys(variable)

        from scipy.interpolate import griddata

        # Get window extent:
        hmin = self.output[hdir].min()
        hmax = self.output[hdir].max()
        vmin = self.output[vdir].min()
        vmax = self.output[vdir].max()

        # Create the original coordinates array:
        coord = np.column_stack((self.output[hdir].to_numpy(), self.output[vdir].to_numpy()))

        # Create the coordinate arrays:
        h_n = np.linspace(hmin, hmax, n_elements)
        v_n = np.linspace(vmin, vmax, n_elements)

        # Create the meshes
        H, V = np.meshgrid(h_n, v_n)

        # Interpolate the variable:
        # if type(variable) == str:
        interpolated = griddata(coord, variable, (H, V), method='linear')
        # else:
        # interpolated[i_var] = griddata(coord, i_var, (H, V), method='linear')
        return H, V, interpolated

    def set_window(self, hmin, hmax, vmin, vmax, window_dirs=('x', 'y')):
        """
        Function to limit the domain to a rectangular window.
        Arguments:
            > Window domain: [hmin, hmax, vmin, vmax]

         Returns:
            Recreated output dictionary with limited mesh size to the specified domain

         Example use:
            model = uw_model('path/to/model')
            model.set_current_ts(time)
            model.get_material()
            model.set_window([0, 100, 0, 1000])

        """
        # TODO: Add polygon support and 3D
        # if self.dim != 2:
        #     raise TypeError('Currently not working for 3D!')

        # Find the specific mesh domain for this window:
        if window_dirs is None:
            window_dirs = ['x', 'y']
        window_bool = np.all(
            [self.output[window_dirs[0]] > hmin,
             self.output[window_dirs[0]] < hmax,
             self.output[window_dirs[1]] > vmin,
             self.output[window_dirs[1]] < vmax],
            axis=0
        )
        window_index = self.output[window_bool].index

        # Recreate the output dictionary:
        for key in self.output:
            self.output[key] = self.output[key].iloc[window_index].reset_index(drop=True)

        # Remove the nans in the dataset:
        self.output = self.output.dropna()

    def limit_by_index(self, index):

        # Recreate the output dictionary:
        for key in self.output:
            self.output[key] = self.output[key].iloc[index].reset_index(drop=True)

    def remove_background(self, bg_phase=1):
        # Background is generally MI = min(MI)
        mi = self.output.mat.unique()
        mi.sort()

        # Drop the rows where this thing is found:
        bg_index = self.output[self.output.mat < mi[mi > bg_phase + 1][0]].index

        # Recreate the output dictionary:
        for key in self.output:
            self.output[key] = self.output[key].drop(bg_index).reset_index(drop=True)

    def extract_by_material(self, mat_index):

        if type(mat_index) == int or type(mat_index) == float:
            # Extract only one material index:
            mat_index = float(mat_index)

            # Limit the output:
            mesh_sorted = self.output.loc[self.output.mat == mat_index]

            # If empty:
            if mesh_sorted.shape[0] == 0:
                raise Exception('Invalid material submitted, check material DB to see if it exists.')

            # Recreate the output files, resetting the index:
            for key in self.output:
                self.output[key] = self.output[key].iloc[mesh_sorted.index]. \
                    reset_index(drop=True)

        elif type(mat_index) == list or type(mat_index) == np.ndarray:

            # Extract by subset
            temp_bool = [self.output.mat.values == float(x) for x in mat_index]

            # Find correct locations:
            temp_bool = sum(temp_bool)  # Any area with zero is not important

            # Get the index array
            index_bool = temp_bool > 0

            # Recreate the output dictionary:
            for key in self.output:
                self.output[key] = self.output[key].iloc[index_bool]. \
                    reset_index(drop=True)

    def remove_slices(self):
        """
        Remove the slices made previously and remake the outputs.

        """

        # Restore the original output:
        # try:
        #     self.output = self._starting_output.copy()
        # except:
        self.set_current_ts(self.current_step)

    def set_slice(self, direction, value=0, n_slices=None, find_closest=True, save=False):
        """
        Creates a slice according to the user specified parameters.
        """

        # This makes unlimited slices of the model. Use with care
        if np.all(direction != 'x' and direction != 'y' and direction != 'z'):
            raise Exception('The slice direction must be: ''x'', ''y'' or ''z''!')

        # Save the original dataframe in a hidden variable:
        # self._starting_output = self.output.copy()

        if save:
            # This would be useful to redo any slicing destroyed by a function or other
            self.performed_slices.append({'direction': direction,
                                          'value': value,
                                          'n_slices': n_slices,
                                          'find_closest': find_closest},
                                         )

        if not n_slices:

            # If the rounder is disables
            if not find_closest:
                # Limit the mesh:
                mesh_sorted = self.output.loc[self.output[direction] == value]

                # If empty:
                if mesh_sorted.shape[0] == 0:
                    raise Exception('Invalid slice value, check mesh dataframe for possible slice index.')

                # Recreate the output files, resetting the index:
                for key in self.output:
                    self.output[key] = self.output[key].iloc[mesh_sorted.index].reset_index(drop=True)
            else:
                # If the rounder is on:

                # get the deltas
                mesh_delta = self.output[direction].copy() - value

                # get the index for the closest value:
                depth_id = mesh_delta.abs().sort_values().index[0]

                # create the slice IDs
                slice_id = self.output[self.output[direction] == self.output[direction].iloc[depth_id]]

                # recreate the domain
                for key in self.output:
                    self.output[key] = self.output[key].iloc[slice_id.index].reset_index(drop=True)

        if n_slices:
            if n_slices > self.res[direction]:
                raise ValueError('More slices than amount of rows along direction.')

            # Make sure this is an integer
            n_slices = int(n_slices)

            # Get the possible values for the direction:
            possible = self.output[direction].unique()
            direction_values = np.linspace(0, len(possible) - 1, n_slices, dtype=int)

            self.slice_info = pd.DataFrame(
                data={'slice_id': range(n_slices), 'slice_value': possible[direction_values]})

            # Extract by subset
            temp_bool = [self.output[direction].values == float(x) for x in possible[direction_values]]

            # Find correct locations:
            temp_bool = sum(temp_bool)  # Any area with zero is not important

            # Get the index array
            index_bool = temp_bool > 0

            #             Recreate the output dictionary:
            for key in self.output:
                self.output[key] = self.output[key].iloc[index_bool]. \
                    reset_index(drop=True)

            # Extract by subset for slicing IDS
            temp_bool = [self.output[direction].values == float(x) for x in possible[direction_values]]
            temp_ids = np.ones(np.array(temp_bool)[0].shape) * 1e3

            for n_slice, index in zip(range(n_slices), temp_bool):
                temp_ids[index] = int(n_slice)

            # Save the ID
            self.output['slice_id'] = temp_ids

        # Regardless of the process, get rid of the nans:
        self.output = self.output.dropna()

    def correct_depth(self, vertical_direction='y'):
        """
        Set the correct vertical direction with depth increasing from the surface.
        Defaults to vertical 'y'.

        Parameters:
           > vertical_direction: single character string with the correct direction

        """
        self.output[vertical_direction] = np.abs(
            self.output[vertical_direction] - self.output[vertical_direction].max())

    #################################################
    #             SUBDUCTION FUNCTIONS              #
    #################################################


class SubductionModel(ModelProcessing):
    def __init__(self, model_dir, horizontal_direction='x',
                 vertical_direction='y', surface_value=0, **kwargs):
        # Initiate the uwobject
        super().__init__(model_dir=model_dir, **kwargs)

        self.horizontal_direction = horizontal_direction
        self.vertical_direction = vertical_direction

        # Correct the depth scale from uw models:
        if self.loader == 'uw':
            self.correct_depth(vertical_direction=vertical_direction)

        # Find the closest surface value possible:
        self.surface_value = get_closest(self.output[vertical_direction], surface_value)

        # Testing if this works
        if self.loader == 'uw':
            # Detect the trench position
            self.trench = self.find_trench()

    def find_trench(self, filter=True):  # , horizontal_plane='xz', override_dim_check=False
        """
        Function that returns the surface position of the subduction trench, following the
        minimum divergence method.


        TODO: 3D

        Returns:
            2D: Horizontal position of the trench
            3D: Coordinate array for a line that represents the trench along the horizontal plane
        """
        # Check for dimensions
        # if self.dim == 2:
        # Get the vertical coordinates
        hdir = self.output[self.horizontal_direction]
        vdir = self.output[self.vertical_direction]

        # Get a surface slice
        surface_index = vdir[vdir == self.surface_value].index

        # Get velocity fields
        # condition = False
        #         # while not condition:
        #         #     try:
        # If this is being called by an external object, try and detect the velocities
        vx = self.output.vx.iloc[surface_index].to_numpy()
        #     condition = True
        # except AttributeError:
        #     # If this is the first loading of the SubductionModel object or the velocities aren't present
        #     self.get_velocity()

        # Extract just the vertical velocity
        # vy = self.output.vy.iloc[surface_index].to_numpy()

        # Calculate the fields 1st derivative
        dvx = np.gradient(vx)
        dx = np.gradient(hdir[surface_index])

        # Calculate divergence (i.e. scalar change of a vector field)
        div_v = dvx / dx

        if filter:
            div_v = div_v[30:-30]

            # Store the trench id:
            trench_id = div_v == min(div_v)

            trench_id = np.array(trench_id)

            trench_id = np.pad(trench_id, 30, mode='constant', constant_values=0)
            # trench_id = np.argmax(trench_id == 1) + 30

        else:

            # Store the trench id:
            trench_id = div_v == min(div_v)

        return float(hdir[surface_index][trench_id])

        # return trench_id
        # elif self.ndims == 3:

    def get_polarity(self, op_material=4, plate_thickness=100., horizontal_plane='xz', trench_direction='z',
                     get_depth=False):
        """
        Function for finding the overriding plate at a critical depth. This depth is 25% deeper than the expected
        thickness.

         Parameters:
            > uw_object: an object created with the uw_model script, loaded with timestep, mesh and material.
            > op_material: the ID or range of IDS for the overriding plate crust.
            > plate_thickness: self-explanatory, maximum expected thickness for the lithosphere in km
            > horizontal_plane: indicate the horizontal plane directions, by default 'xy'.
                                  Options: 'xy', 'yz', 'xz'
            > trench_direction: indicate the along trench direction, by default 'z'.
                                  Options: 'x', 'y', 'z'

         Returns:
            New dataframe under model.polarity.
            model.polarity with two columns: along trench axis positions and polarity state.
            Zero (0) represents normal (i.e. initial polarity) while one (1) represents a reversed state.


         Example use:
            model = uw_model('path/to/model')
            model.set_current_ts(time)
            model.get_mesh()
            model.get_material()
            model.get_polarity()
            :param plate_thickness:
        """
        # Catch a few errors:
        if type(horizontal_plane) != str:
            raise TypeError('Plane must be a string!')

        if len(horizontal_plane) != 2:
            raise ValueError('Plane can only contain two letters!')

        if len(trench_direction) != 1:
            raise ValueError('Trench direction is a single letter!')

        # ====================================== CHECK VALIDITY ======================================

        # Ensure the strings are correctly formatted.
        horizontal_plane = "".join(sorted(horizontal_plane.lower()))  # Correctly sorted and in lower case.
        trench_direction = trench_direction.lower()

        # Check if the plane is valid:
        valid_planes = ['xy', 'yz', 'xz']
        check = np.sum([sorted(horizontal_plane) == sorted(valid) for valid in valid_planes])

        if check == 0:
            raise ValueError('Plane is invalid. Please try a combination of ''x'', ''y'' and ''z''.')

        # Check the plane direction:
        slice_direction = 'xyz'

        for char in horizontal_plane:
            slice_direction = slice_direction.replace(char, '')

        # Check if the direction of the trench is valid:
        valid_direction = ['x', 'y', 'z']
        check = np.sum([trench_direction == valid for valid in valid_direction])
        #
        # # Get the vertical direction:
        # directions_in_plane = [letter for letter in horizontal_plane]
        # vertical_direction = [letter for letter in valid_direction if letter not in directions_in_plane] # Whatever
        # # is not on the plane

        if check == 0:
            raise ValueError('Trench is invalid. Please try ''x'', ''y'' or ''z''.')

        # ================================ DETECT THE POLARITY ========================================

        # Remove any slices:
        self.remove_slices()

        # Set the critical depth:
        if self.loader == 'uw':
            critical_depth = 1.25 * plate_thickness * 1e3
            # Create a slice at that depth:
            self.set_slice(slice_direction, value=self.output[slice_direction].max() - critical_depth,
                           find_closest=True)

        else:
            critical_depth = -1.25 * plate_thickness
            # Create a slice at that depth:
            self.set_slice(slice_direction, value=critical_depth,
                           find_closest=True)

        # Create a database just for the next operations, saves on memory and code:
        reversed_index = self.output.mat[self.output.mat.round() == op_material].index.to_numpy()

        # Detect along trench direction where it is reversed:
        trench_dir_reverse = self.output[trench_direction].loc[reversed_index].unique()

        # Remove any slices:
        self.remove_slices()

        # Create a zeros array, each zero will represent the normal polarity
        polarity = pd.DataFrame(data=np.array([self.output[trench_direction].to_numpy(),
                                               np.zeros(self.output.x.shape)]).T,
                                columns=(trench_direction, 'state'))

        # Check all locations where trench direction reversed is found:
        _, _, reversed_index = np.intersect1d(trench_dir_reverse,
                                              self.output[trench_direction].to_numpy(),
                                              return_indices=True)

        # This only adds a zero to a single value of that trench_direction value:
        polarity.loc[reversed_index, 'state'] = 1

        # Copy those values for all trench_direction values:
        for td in trench_dir_reverse:
            polarity.state[polarity[trench_direction] == td] = 1

        # Add polarity to all dataframes now:
        self.output['polarity'] = polarity.state.copy()

        # Check slices that were made before:
        needed_slices = self.performed_slices.copy()

        # Remake the ones deleted:
        for slices in needed_slices:
            # print(f'Making slice: {slices}')
            self.set_slice(**slices)

        if get_depth:
            return critical_depth


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Preparar os dois loaders:
    uw_model = ModelProcessing(model_dir='F:\\NoPlateauSubduction\\model_results\\10OP_30DP',
                               scf=1e22, ts=300)

    # lm_model = SubductionModel(model_dir='Z:\\PlateauCollision3D_LM\\plateau_size\\_M_D70_O70',
    #                            loader='lamem',
    #                            vtk_name='Caribbean_v16', ts=200)
    #
    # # FLip the x array
    # lm_model.output.x *= -1
    #
    # # Set slice at the surface:
    # uw_model.set_slice(direction='z', value=0, find_closest=True)
    # lm_model.set_slice(direction='y', value=0, find_closest=True)
    #
    # # Get the trenches
    # uw_trench = uw_model.find_trench()
    # lm_trench = lm_model.find_trench(filter=False)
    #
    # # Interpolate
    # x_uw, y_uw, mat_uw = uw_model.reinterpolate_window(variable=uw_model.output.eta, hdir='x', vdir='y')
    # x_lm, y_lm, mat_lm = lm_model.reinterpolate_window(variable=lm_model.output.eta, hdir='x', vdir='z')
    #
    # # Testar se funcionam os dois:
    # fig, ax = plt.subplots(nrows=2, figsize=[6, 8])
    #
    # # Start plotting
    # uw_plot = ax[1].pcolormesh(x_uw / 1e3, y_uw / 1e3, mat_uw, cmap='coolwarm', shading='auto')
    # ax[1].set_xlabel('x')
    # ax[1].set_ylabel('y')
    # ax[1].set_title('UwLoader')
    # ax[1].axvline(uw_trench / 1e3)
    #
    # lm_plot = ax[0].pcolormesh(x_lm, y_lm, mat_lm, cmap='coolwarm', shading='auto')
    # ax[0].set_xlabel('x')
    # ax[0].set_ylabel('y')
    # ax[0].set_title('LaMEMLoader')
    # ax[0].axvline(lm_trench)
    #
    # plt.colorbar(lm_plot, ax=ax[0])
    # plt.colorbar(uw_plot, ax=ax[1])
    # fig.tight_layout()
