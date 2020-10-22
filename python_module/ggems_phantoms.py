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

        ggems_lib.set_voxelized_phantom_file_ggems_phantom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        ggems_lib.set_voxelized_phantom_file_ggems_phantom.restype = ctypes.c_void_p

        # ggems_lib.set_position_ggems_phantom.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_char_p]
        # ggems_lib.set_position_ggems_phantom.restype = ctypes.c_void_p

        self.obj = ggems_lib.create_ggems_voxelized_phantom(voxelized_phantom_name.encode('ASCII'))

    def set_voxelized_phantom(self, phantom_filename, range_data_filename):
        ggems_lib.set_voxelized_phantom_file_ggems_phantom(self.obj, phantom_filename.encode('ASCII'), range_data_filename.encode('ASCII'))