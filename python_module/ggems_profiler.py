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

class GGEMSProfilerManager(object):
    """Get the C++ singleton and print summary about profiler
    """
    def __init__(self):
        ggems_lib.get_instance_profiler_manager.restype = ctypes.c_void_p

        ggems_lib.print_summary_profiler_manager.argtypes = [ctypes.c_void_p]
        ggems_lib.print_summary_profiler_manager.restype = ctypes.c_void_p

        self.obj = ggems_lib.get_instance_profiler_manager()

    def print_summary_profile(self):
        ggems_lib.print_summary_profiler_manager(self.obj)

