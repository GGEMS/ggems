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

class GGEMSDosimetryCalculator(object):
    """Class for dosimetry computation
    """
    def __init__(self, voxelized_phantom_name):
        ggems_lib.create_ggems_dosimetry_calculator.restype = ctypes.c_void_p

        ggems_lib.set_dosel_size_dosimetry_calculator.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_dosel_size_dosimetry_calculator.restype = ctypes.c_void_p

        ggems_lib.set_dose_output_dosimetry_calculator.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_dose_output_dosimetry_calculator.restype = ctypes.c_void_p

        ggems_lib.dose_photon_tracking_dosimetry_calculator.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.dose_photon_tracking_dosimetry_calculator.restype = ctypes.c_void_p

        ggems_lib.dose_edep_dosimetry_calculator.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.dose_edep_dosimetry_calculator.restype = ctypes.c_void_p

        ggems_lib.dose_hit_dosimetry_calculator.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.dose_hit_dosimetry_calculator.restype = ctypes.c_void_p

        ggems_lib.dose_edep_squared_dosimetry_calculator.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.dose_edep_squared_dosimetry_calculator.restype = ctypes.c_void_p

        ggems_lib.dose_uncertainty_dosimetry_calculator.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.dose_uncertainty_dosimetry_calculator.restype = ctypes.c_void_p

        ggems_lib.delete_dosimetry_calculator.argtypes = [ctypes.c_void_p]
        ggems_lib.delete_dosimetry_calculator.restype = ctypes.c_void_p

        ggems_lib.scale_factor_dosimetry_calculator.argtypes = [ctypes.c_void_p, ctypes.c_float]
        ggems_lib.scale_factor_dosimetry_calculator.restype = ctypes.c_void_p

        ggems_lib.minimum_density_dosimetry_calculator.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.minimum_density_dosimetry_calculator.restype = ctypes.c_void_p

        ggems_lib.water_reference_dosimetry_calculator.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.water_reference_dosimetry_calculator.restype = ctypes.c_void_p

        self.obj = ggems_lib.create_ggems_dosimetry_calculator(voxelized_phantom_name.encode('ASCII'))

    def set_dosel_size(self, dose_x, dose_y, dose_z, unit):
        ggems_lib.set_dosel_size_dosimetry_calculator(self.obj, dose_x, dose_y, dose_z, unit.encode('ASCII'))

    def delete(self):
        ggems_lib.delete_dosimetry_calculator(self.obj)

    def set_output_basename(self, output):
        ggems_lib.set_dose_output_dosimetry_calculator(self.obj, output.encode('ASCII'))

    def photon_tracking(self, activate):
        ggems_lib.dose_photon_tracking_dosimetry_calculator(self.obj, activate)

    def edep(self, activate):
        ggems_lib.dose_edep_dosimetry_calculator(self.obj, activate)

    def hit(self, activate):
        ggems_lib.dose_hit_dosimetry_calculator(self.obj, activate)

    def edep_squared(self, activate):
        ggems_lib.dose_edep_squared_dosimetry_calculator(self.obj, activate)

    def uncertainty(self, activate):
        ggems_lib.dose_uncertainty_dosimetry_calculator(self.obj, activate)

    def scale_factor(self, scale):
        ggems_lib.scale_factor_dosimetry_calculator(self.obj, scale)

    def water_reference(self, activate):
        ggems_lib.water_reference_dosimetry_calculator(self.obj, activate)

    def minimum_density(self, density, unit):
        ggems_lib.minimum_density_dosimetry_calculator(self.obj, density, unit.encode('ASCII'))
