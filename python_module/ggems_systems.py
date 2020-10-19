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

class GGEMSPhantom(object):
    """Class for phantom for GGEMS simulation
    """
    def __init__(self, phantom_name, phantom_type):
        ggems_lib.create_ggems_phantom.restype = ctypes.c_void_p

        ggems_lib.set_phantom_file_ggems_phantom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        ggems_lib.set_phantom_file_ggems_phantom.restype = ctypes.c_void_p

        ggems_lib.set_position_ggems_phantom.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_position_ggems_phantom.restype = ctypes.c_void_p

        self.obj = ggems_lib.create_ggems_phantom(phantom_name.encode('ASCII'), phantom_type.encode('ASCII'))

    def set_phantom_image(self, phantom_filename, range_data_filename):
        ggems_lib.set_phantom_file_ggems_phantom(self.obj, phantom_filename.encode('ASCII'), range_data_filename.encode('ASCII'))


class GGEMSCTSystem(object):
    """Class for CT/CBCT for GGEMS simulation
    """
    def __init__(self, ct_system_name):
        ggems_lib.create_ggems_ct_system.restype = ctypes.c_void_p

        self.obj = ggems_lib.create_ggems_ct_system(ct_system_name.encode('ASCII'))