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