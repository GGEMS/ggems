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

class GGEMSOpenCLManager(object):
    """Get the OpenCL C++ singleton and print infos or managing it
    """
    def __init__(self):
        ggems_lib.get_instance_ggems_opencl_manager.restype = ctypes.c_void_p

        ggems_lib.print_infos_opencl_manager.argtypes = [ctypes.c_void_p]
        ggems_lib.print_infos_opencl_manager.restype = ctypes.c_void_p

        ggems_lib.set_context_index_ggems_opencl_manager.argtypes = [ctypes.c_void_p, ctypes.c_int]
        ggems_lib.set_context_index_ggems_opencl_manager.restype = ctypes.c_void_p

        self.obj = ggems_lib.get_instance_ggems_opencl_manager()

    def print_infos(self):
        ggems_lib.print_infos_opencl_manager(self.obj)

    def set_context_index(self, context_id):
        ggems_lib.set_context_index_ggems_opencl_manager(self.obj, context_id)
