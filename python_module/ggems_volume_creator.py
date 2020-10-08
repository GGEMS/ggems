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

class GGEMSVolumeCreatorManager(object):
    """Get Volume Creator Manager to convert analytical volume to voxelized volume
    """
    def __init__(self):
        ggems_lib.get_instance_volume_creator_manager.restype = ctypes.c_void_p

        ggems_lib.set_volume_dimension_volume_creator_manager.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        ggems_lib.set_volume_dimension_volume_creator_manager.restype = ctypes.c_void_p

        ggems_lib.set_element_sizes_volume_creator_manager.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_element_sizes_volume_creator_manager.restype = ctypes.c_void_p

        ggems_lib.set_output_image_filename_volume_creator_manager.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_output_image_filename_volume_creator_manager.restype = ctypes.c_void_p

        ggems_lib.set_output_range_to_material_filename_volume_creator_manager.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_output_range_to_material_filename_volume_creator_manager.restype = ctypes.c_void_p

        ggems_lib.initialize_volume_creator_manager.argtypes = [ctypes.c_void_p]
        ggems_lib.initialize_volume_creator_manager.restype = ctypes.c_void_p

        ggems_lib.write_volume_creator_manager.argtypes = [ctypes.c_void_p]
        ggems_lib.write_volume_creator_manager.restype = ctypes.c_void_p

        ggems_lib.set_material_volume_creator_manager.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_material_volume_creator_manager.restype = ctypes.c_void_p

        ggems_lib.set_data_type_volume_creator_manager.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_data_type_volume_creator_manager.restype = ctypes.c_void_p

        self.obj = ggems_lib.get_instance_volume_creator_manager()

    def set_dimensions(self, width, height, depth):
        ggems_lib.set_volume_dimension_volume_creator_manager(self.obj, width, height, depth)

    def set_element_sizes(self, width, height, depth, unit):
        ggems_lib.set_element_sizes_volume_creator_manager(self.obj, width, height, depth, unit.encode('ASCII'))

    def set_output(self, output):
        ggems_lib.set_output_image_filename_volume_creator_manager(self.obj, output.encode('ASCII'))

    def set_range_output(self, output):
        ggems_lib.set_output_range_to_material_filename_volume_creator_manager(self.obj, output.encode('ASCII'))

    def initialize(self):
        ggems_lib.initialize_volume_creator_manager(self.obj)

    def write(self):
        ggems_lib.write_volume_creator_manager(self.obj)

    def set_material(self, material):
        ggems_lib.set_material_volume_creator_manager(self.obj, material.encode('ASCII'))

    def set_data_type(self, data_type):
        ggems_lib.set_data_type_volume_creator_manager(self.obj, data_type.encode('ASCII'))


class GGEMSTube(object):
    """Build a solid tube analytical phantom
    """
    def __init__(self):
        ggems_lib.create_tube.restype = ctypes.c_void_p

        ggems_lib.delete_tube.argtypes = [ctypes.c_void_p]
        ggems_lib.delete_tube.restype = ctypes.c_void_p

        ggems_lib.set_height_tube.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_height_tube.restype = ctypes.c_void_p

        ggems_lib.set_radius_tube.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_radius_tube.restype = ctypes.c_void_p

        ggems_lib.set_position_tube.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_position_tube.restype = ctypes.c_void_p

        ggems_lib.set_label_value_tube.argtypes = [ctypes.c_void_p, ctypes.c_float]
        ggems_lib.set_label_value_tube.restype = ctypes.c_void_p

        ggems_lib.set_material_tube.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_material_tube.restype = ctypes.c_void_p

        ggems_lib.initialize_tube.argtypes = [ctypes.c_void_p]
        ggems_lib.initialize_tube.restype = ctypes.c_void_p

        ggems_lib.draw_tube.argtypes = [ctypes.c_void_p]
        ggems_lib.draw_tube.restype = ctypes.c_void_p

        self.obj = ggems_lib.create_tube()

    def delete(self):
        ggems_lib.delete_tube(self.obj)

    def set_height(self, height, unit):
        ggems_lib.set_height_tube(self.obj, height, unit.encode('ASCII'))

    def set_radius(self, radius, unit):
        ggems_lib.set_radius_tube(self.obj, radius, unit.encode('ASCII'))

    def set_label_value(self, label_value):
        ggems_lib.set_label_value_tube(self.obj, label_value)

    def set_position(self, pos_x, pos_y, pos_z, unit):
        ggems_lib.set_position_tube(self.obj, pos_x, pos_y, pos_z, unit.encode('ASCII'))

    def set_material(self, material):
        ggems_lib.set_material_tube(self.obj, material.encode('ASCII'))

    def initialize(self):
        ggems_lib.initialize_tube(self.obj)

    def draw(self):
        ggems_lib.draw_tube(self.obj)


class GGEMSBox(object):
    """Build a solid box analytical phantom
    """
    def __init__(self):
        ggems_lib.create_box.restype = ctypes.c_void_p

        ggems_lib.delete_box.argtypes = [ctypes.c_void_p]
        ggems_lib.delete_box.restype = ctypes.c_void_p

        ggems_lib.set_height_box.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_height_box.restype = ctypes.c_void_p

        ggems_lib.set_width_box.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_width_box.restype = ctypes.c_void_p

        ggems_lib.set_depth_box.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_depth_box.restype = ctypes.c_void_p

        ggems_lib.set_position_box.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_position_box.restype = ctypes.c_void_p

        ggems_lib.set_label_value_box.argtypes = [ctypes.c_void_p, ctypes.c_float]
        ggems_lib.set_label_value_box.restype = ctypes.c_void_p

        ggems_lib.set_material_box.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_material_box.restype = ctypes.c_void_p

        ggems_lib.initialize_box.argtypes = [ctypes.c_void_p]
        ggems_lib.initialize_box.restype = ctypes.c_void_p

        ggems_lib.draw_box.argtypes = [ctypes.c_void_p]
        ggems_lib.draw_box.restype = ctypes.c_void_p

        self.obj = ggems_lib.create_box()

    def delete(self):
        ggems_lib.delete_box(self.obj)

    def set_height(self, height, unit):
        ggems_lib.set_height_box(self.obj, height, unit.encode('ASCII'))

    def set_width(self, width, unit):
        ggems_lib.set_width_box(self.obj, width, unit.encode('ASCII'))

    def set_depth(self, depth, unit):
        ggems_lib.set_depth_box(self.obj, depth, unit.encode('ASCII'))

    def set_label_value(self, label_value):
        ggems_lib.set_label_value_box(self.obj, label_value)

    def set_position(self, pos_x, pos_y, pos_z, unit):
        ggems_lib.set_position_box(self.obj, pos_x, pos_y, pos_z, unit.encode('ASCII'))

    def set_material(self, material):
        ggems_lib.set_material_box(self.obj, material.encode('ASCII'))

    def initialize(self):
        ggems_lib.initialize_box(self.obj)

    def draw(self):
        ggems_lib.draw_box(self.obj)