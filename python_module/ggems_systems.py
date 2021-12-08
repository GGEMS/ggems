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

        ggems_lib.set_number_of_modules_ggems_ct_system.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
        ggems_lib.set_number_of_modules_ggems_ct_system.restype = ctypes.c_void_p

        ggems_lib.set_ct_system_type_ggems_ct_system.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_ct_system_type_ggems_ct_system.restype = ctypes.c_void_p

        ggems_lib.set_number_of_detection_elements_ggems_ct_system.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
        ggems_lib.set_number_of_detection_elements_ggems_ct_system.restype = ctypes.c_void_p

        ggems_lib.set_size_of_detection_elements_ggems_ct_system.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_size_of_detection_elements_ggems_ct_system.restype = ctypes.c_void_p

        ggems_lib.set_material_name_ggems_ct_system.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_material_name_ggems_ct_system.restype = ctypes.c_void_p

        ggems_lib.set_source_isocenter_distance_ggems_ct_system.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_source_isocenter_distance_ggems_ct_system.restype = ctypes.c_void_p

        ggems_lib.set_source_detector_distance_ggems_ct_system.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_source_detector_distance_ggems_ct_system.restype = ctypes.c_void_p

        ggems_lib.set_rotation_ggems_ct_system.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_rotation_ggems_ct_system.restype = ctypes.c_void_p

        ggems_lib.set_material_visible_ggems_ct_system.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool]
        ggems_lib.set_material_visible_ggems_ct_system.restype = ctypes.c_void_p

        ggems_lib.set_material_color_ggems_ct_system.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_ubyte]
        ggems_lib.set_material_color_ggems_ct_system.restype = ctypes.c_void_p

        ggems_lib.set_visible_ggems_ct_system.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.set_visible_ggems_ct_system.restype = ctypes.c_void_p

        ggems_lib.set_threshold_ggems_ct_system.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_threshold_ggems_ct_system.restype = ctypes.c_void_p

        ggems_lib.set_save_ggems_ct_system.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_save_ggems_ct_system.restype = ctypes.c_void_p

        ggems_lib.store_scatter_ggems_ct_system.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.store_scatter_ggems_ct_system.restype = ctypes.c_void_p

        self.obj = ggems_lib.create_ggems_ct_system(ct_system_name.encode('ASCII'))

    def set_number_of_modules(self, module_x, module_y):
        ggems_lib.set_number_of_modules_ggems_ct_system(self.obj, module_x, module_y)

    def set_ct_type(self, ct_system_type):
        ggems_lib.set_ct_system_type_ggems_ct_system(self.obj, ct_system_type.encode('ASCII'))

    def set_number_of_detection_elements(self, element_x, element_y, element_z):
        ggems_lib.set_number_of_detection_elements_ggems_ct_system(self.obj, element_x, element_y, element_z)

    def set_size_of_detection_elements(self, size_x, size_y, size_z, unit):
        ggems_lib.set_size_of_detection_elements_ggems_ct_system(self.obj, size_x, size_y, size_z, unit.encode('ASCII'))

    def set_material(self, material_name):
        ggems_lib.set_material_name_ggems_ct_system(self.obj, material_name.encode('ASCII'))

    def set_source_detector_distance(self, sdd, unit):
        ggems_lib.set_source_detector_distance_ggems_ct_system(self.obj, sdd, unit.encode('ASCII'))

    def set_source_isocenter_distance(self, sid, unit):
        ggems_lib.set_source_isocenter_distance_ggems_ct_system(self.obj, sid, unit.encode('ASCII'))

    def set_rotation(self, rx, ry, rz, unit):
        ggems_lib.set_rotation_ggems_ct_system(self.obj, rx, ry, rz, unit.encode('ASCII'))

    def set_threshold(self, threshold, unit):
        ggems_lib.set_threshold_ggems_ct_system(self.obj, threshold, unit.encode('ASCII'))

    def set_material_visible(self, material_name, flag):
        ggems_lib.set_material_visible_ggems_ct_system(self.obj, material_name.encode('ASCII'), flag)

    def set_material_color(self, material_name, red, green, blue):
        ggems_lib.set_material_color_ggems_ct_system(self.obj, material_name.encode('ASCII'), red, green, blue)

    def set_visible(self, flag):
        ggems_lib.set_visible_ggems_ct_system(self.obj, flag)

    def save(self, basename):
        ggems_lib.set_save_ggems_ct_system(self.obj, basename.encode('ASCII'))

    def store_scatter(self, flag):
        ggems_lib.store_scatter_ggems_ct_system(self.obj, flag)
