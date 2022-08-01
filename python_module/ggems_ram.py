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

class GGEMSRAMManager(object):
    """Get the C++ singleton and print infos about RAM memory
    """
    def __init__(self):
        ggems_lib.get_instance_ggems_ram_manager.restype = ctypes.c_void_p

        ggems_lib.print_infos_ram_manager.argtypes = [ctypes.c_void_p]
        ggems_lib.print_infos_ram_manager.restype = ctypes.c_void_p

        self.obj = ggems_lib.get_instance_ggems_ram_manager()

    def print_infos(self):
        ggems_lib.print_infos_ram_manager(self.obj)

