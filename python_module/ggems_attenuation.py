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

class GGEMSAttenuations(object):
    """ Class handling attenuations in GGEMS
    """
    def __init__(self, materials, cross_sections):
        ggems_lib.create_ggems_attenuations.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        ggems_lib.create_ggems_attenuations.restype = ctypes.c_void_p

        ggems_lib.initialize_ggems_attenuations.argtypes = [ctypes.c_void_p]
        ggems_lib.initialize_ggems_attenuations.restype = ctypes.c_void_p

        ggems_lib.get_mu_ggems_attenuations.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.get_mu_ggems_attenuations.restype = ctypes.c_float

        ggems_lib.get_mu_en_ggems_attenuations.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.get_mu_en_ggems_attenuations.restype = ctypes.c_float

        ggems_lib.clean_ggems_attenuations.argtypes = [ctypes.c_void_p]
        ggems_lib.clean_ggems_attenuations.restype = ctypes.c_void_p

        self.obj = ggems_lib.create_ggems_attenuations(materials.obj, cross_sections.obj)

    def initialize(self):
        ggems_lib.initialize_ggems_attenuations(self.obj)

    def clean(self):
        ggems_lib.clean_ggems_attenuations(self.obj)

    def get_mu(self, material_name, energy, unit):
        return ggems_lib.get_mu_ggems_attenuations(self.obj, material_name.encode('ASCII'), energy, unit.encode('ASCII'))

    def get_mu_en(self, material_name, energy, unit):
        return ggems_lib.get_mu_en_ggems_attenuations(self.obj, material_name.encode('ASCII'), energy, unit.encode('ASCII'))
