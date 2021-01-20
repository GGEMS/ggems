# ************************************************************************
# * This file is part of GGEMS.                                          *
# *                                                                      *
# * GGEMS is free software: you can redistribute it and/or modify        *
# * it under the terms of the GNU General Public License as published by *
# * the Free Software Foundation, either version 3 of the License, or    *
# * (at your option) any later version.                                  *
# *                                                                      *
# * GGEMS is distributed in the hope that it will be useful,             *
# * but WITHOUT ANY WARRANTY; without even the implied warranty of       *
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        *
# * GNU General Public License for more details.                         *
# *                                                                      *
# * You should have received a copy of the GNU General Public License    *
# * along with GGEMS.  If not, see <https://www.gnu.org/licenses/>.      *
# *                                                                      *
# ************************************************************************

from ggems_lib import *

class GGEMSVoxelizedPhantom(object):
    """Class for voxelized phantom for GGEMS simulation
    """
    def __init__(self, voxelized_phantom_name):
        ggems_lib.create_ggems_voxelized_phantom.restype = ctypes.c_void_p

        ggems_lib.set_phantom_file_ggems_voxelized_phantom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        ggems_lib.set_phantom_file_ggems_voxelized_phantom.restype = ctypes.c_void_p

        ggems_lib.set_position_ggems_voxelized_phantom.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_position_ggems_voxelized_phantom.restype = ctypes.c_void_p

        ggems_lib.set_rotation_ggems_voxelized_phantom.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_rotation_ggems_voxelized_phantom.restype = ctypes.c_void_p

        ggems_lib.set_dosimetry_mode_voxelized_phantom.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.set_dosimetry_mode_voxelized_phantom.restype = ctypes.c_void_p

        ggems_lib.set_dosel_size_voxelized_phantom.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_dosel_size_voxelized_phantom.restype = ctypes.c_void_p

        ggems_lib.set_dose_output_voxelized_phantom.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_dose_output_voxelized_phantom.restype = ctypes.c_void_p

        ggems_lib.dose_photon_tracking_voxelized_phantom.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.dose_photon_tracking_voxelized_phantom.restype = ctypes.c_void_p

        self.obj = ggems_lib.create_ggems_voxelized_phantom(voxelized_phantom_name.encode('ASCII'))

    def set_phantom(self, phantom_filename, range_data_filename):
        ggems_lib.set_phantom_file_ggems_voxelized_phantom(self.obj, phantom_filename.encode('ASCII'), range_data_filename.encode('ASCII'))

    def set_position(self, pos_x, pos_y, pos_z, unit):
        ggems_lib.set_position_ggems_voxelized_phantom(self.obj, pos_x, pos_y, pos_z, unit.encode('ASCII'))

    def set_rotation(self, rx, ry, rz, unit):
        ggems_lib.set_rotation_ggems_voxelized_phantom(self.obj, rx, ry, rz, unit.encode('ASCII'))

    def set_dosimetry_mode(self, mode):
        ggems_lib.set_dosimetry_mode_voxelized_phantom(self.obj, mode)

    def set_dosel_size(self, dose_x, dose_y, dose_z, unit):
        ggems_lib.set_dosel_size_voxelized_phantom(self.obj, dose_x, dose_y, dose_z, unit.encode('ASCII'))

    def set_dose_output(self, output):
        ggems_lib.set_dose_output_voxelized_phantom(self.obj, output.encode('ASCII'))

    def dose_photon_tracking(self, activate):
        ggems_lib.dose_photon_tracking_voxelized_phantom(self.obj, activate)
