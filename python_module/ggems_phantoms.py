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

from .ggems_lib import *

class GGEMSVoxelizedPhantom(object):
    """Class for voxelized phantom for GGEMS simulation
    """
    def __init__(self, voxelized_phantom_name):
        ggems_lib.create_ggems_voxelized_phantom.restype = ctypes.c_void_p

        ggems_lib.set_phantom_file_ggems_voxelized_phantom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        ggems_lib.set_phantom_file_ggems_voxelized_phantom.restype = ctypes.c_void_p

        ggems_lib.set_position_ggems_voxelized_phantom.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_position_ggems_voxelized_phantom.restype = ctypes.c_void_p

        ggems_lib.set_material_visible_ggems_voxelized_phantom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool]
        ggems_lib.set_material_visible_ggems_voxelized_phantom.restype = ctypes.c_void_p

        ggems_lib.set_material_color_ggems_voxelized_phantom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_ubyte]
        ggems_lib.set_material_color_ggems_voxelized_phantom.restype = ctypes.c_void_p

        ggems_lib.set_visible_ggems_voxelized_phantom.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.set_visible_ggems_voxelized_phantom.restype = ctypes.c_void_p

        ggems_lib.set_material_color_name_ggems_voxelized_phantom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        ggems_lib.set_material_color_name_ggems_voxelized_phantom.restype = ctypes.c_void_p

        ggems_lib.set_rotation_ggems_voxelized_phantom.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_rotation_ggems_voxelized_phantom.restype = ctypes.c_void_p

        self.obj = ggems_lib.create_ggems_voxelized_phantom(voxelized_phantom_name.encode('ASCII'))

    def set_phantom(self, phantom_filename, range_data_filename):
        ggems_lib.set_phantom_file_ggems_voxelized_phantom(self.obj, phantom_filename.encode('ASCII'), range_data_filename.encode('ASCII'))

    def set_material_visible(self, material_name, flag):
        ggems_lib.set_material_visible_ggems_voxelized_phantom(self.obj, material_name.encode('ASCII'), flag)

    def set_visible(self, flag):
        ggems_lib.set_visible_ggems_voxelized_phantom(self.obj, flag)

    def set_material_color(self, material_name, red=0, green=0, blue=0, color_name=''):
        if color_name:
            ggems_lib.set_material_color_name_ggems_voxelized_phantom(self.obj, material_name.encode('ASCII'), color_name.encode('ASCII'))
        else:
            ggems_lib.set_material_color_ggems_voxelized_phantom(self.obj, material_name.encode('ASCII'), red, green, blue)

    def set_position(self, pos_x, pos_y, pos_z, unit):
        ggems_lib.set_position_ggems_voxelized_phantom(self.obj, pos_x, pos_y, pos_z, unit.encode('ASCII'))

    def set_rotation(self, rx, ry, rz, unit):
        ggems_lib.set_rotation_ggems_voxelized_phantom(self.obj, rx, ry, rz, unit.encode('ASCII'))


class GGEMSMeshedPhantom(object):
    """Class for meshed phantom for GGEMS simulation
    """
    def __init__(self, meshed_phantom_name):
        ggems_lib.create_ggems_meshed_phantom.restype = ctypes.c_void_p

        ggems_lib.set_phantom_file_ggems_meshed_phantom.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_phantom_file_ggems_meshed_phantom.restype = ctypes.c_void_p

        self.obj = ggems_lib.create_ggems_meshed_phantom(meshed_phantom_name.encode('ASCII'))

    def set_phantom(self, phantom_filename):
        ggems_lib.set_phantom_file_ggems_meshed_phantom(self.obj, phantom_filename.encode('ASCII'))


class GGEMSWorld(object):
    """Class for world volume for GGEMS simulation
    """
    def __init__(self):
        ggems_lib.create_ggems_world.restype = ctypes.c_void_p

        ggems_lib.set_dimension_ggems_world.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        ggems_lib.set_dimension_ggems_world.restype = ctypes.c_void_p

        ggems_lib.set_size_ggems_world.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_size_ggems_world.restype = ctypes.c_void_p

        ggems_lib.photon_tracking_ggems_world.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.photon_tracking_ggems_world.restype = ctypes.c_void_p

        ggems_lib.energy_tracking_ggems_world.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.energy_tracking_ggems_world.restype = ctypes.c_void_p

        ggems_lib.set_output_ggems_world.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_output_ggems_world.restype = ctypes.c_void_p

        ggems_lib.energy_squared_tracking_ggems_world.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.energy_squared_tracking_ggems_world.restype = ctypes.c_void_p

        ggems_lib.momentum_ggems_world.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.momentum_ggems_world.restype = ctypes.c_void_p

        self.obj = ggems_lib.create_ggems_world()

    def set_dimensions(self, dim_x, dim_y, dim_z):
        ggems_lib.set_dimension_ggems_world(self.obj, dim_x, dim_y, dim_z)

    def set_element_sizes(self, size_x, size_y, size_z, unit):
        ggems_lib.set_size_ggems_world(self.obj, size_x, size_y, size_z, unit.encode('ASCII'))

    def photon_tracking(self, activate):
        ggems_lib.photon_tracking_ggems_world(self.obj, activate)

    def set_output_basename(self, output):
        ggems_lib.set_output_ggems_world(self.obj, output.encode('ASCII'))

    def energy_tracking(self, activate):
        ggems_lib.energy_tracking_ggems_world(self.obj, activate)

    def energy_squared_tracking(self, activate):
        ggems_lib.energy_squared_tracking_ggems_world(self.obj, activate)

    def momentum(self, activate):
        ggems_lib.momentum_ggems_world(self.obj, activate)
