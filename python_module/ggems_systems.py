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

class GGEMSCTSystem(object):
    """Class for CT/CBCT for GGEMS simulation
    """
    def __init__(self, ct_system_name):
        ggems_lib.create_ggems_ct_system.restype = ctypes.c_void_p

        ggems_lib.set_number_of_modules_ggems_ct_system.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint]
        ggems_lib.set_number_of_modules_ggems_ct_system.restype = ctypes.c_void_p

        self.obj = ggems_lib.create_ggems_ct_system(ct_system_name.encode('ASCII'))

    def set_number_of_modules(self, module_x, module_y):
        ggems_lib.set_number_of_modules_ggems_ct_system(self.obj, module_x, module_y)