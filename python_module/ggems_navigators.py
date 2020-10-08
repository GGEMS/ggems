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

class GGEMSVoxelizedNavigator(object):
    """Class for voxelized navigator for GGEMS simulation
    """
    def __init__(self):
        ggems_lib.create_ggems_voxelized_navigator.restype = ctypes.c_void_p

        ggems_lib.set_phantom_name_ggems_voxelized_navigator.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_phantom_name_ggems_voxelized_navigator.restype = ctypes.c_void_p

        ggems_lib.set_phantom_file_ggems_voxelized_navigator.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_phantom_file_ggems_voxelized_navigator.restype = ctypes.c_void_p

        ggems_lib.set_range_to_material_filename_ggems_voxelized_navigator.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_range_to_material_filename_ggems_voxelized_navigator.restype = ctypes.c_void_p

        ggems_lib.set_geometry_tolerance_ggems_voxelized_navigator.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_geometry_tolerance_ggems_voxelized_navigator.restype = ctypes.c_void_p

        ggems_lib.set_position_ggems_voxelized_navigator.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_position_ggems_voxelized_navigator.restype = ctypes.c_void_p

        self.obj = ggems_lib.create_ggems_voxelized_navigator()

    def set_phantom_name(self, name):
        ggems_lib.set_phantom_name_ggems_voxelized_navigator(self.obj, name.encode('ASCII'))

    def set_phantom_image(self, name):
        ggems_lib.set_phantom_file_ggems_voxelized_navigator(self.obj, name.encode('ASCII'))

    def set_range_to_material(self, name):
        ggems_lib.set_range_to_material_filename_ggems_voxelized_navigator(self.obj, name.encode('ASCII'))

    def set_geometry_tolerance(self, distance, unit):
        ggems_lib.set_geometry_tolerance_ggems_voxelized_navigator(self.obj, distance, unit.encode('ASCII'))

    def set_position(self, offset_x, offset_y, offset_z, unit):
        ggems_lib.set_position_ggems_voxelized_navigator(self.obj, offset_x, offset_y, offset_z, unit.encode('ASCII'))