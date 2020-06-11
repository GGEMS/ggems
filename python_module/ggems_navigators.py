from ggems_lib import *


class GGEMSVoxelizedNavigator(object):
    """Class for voxelized navigator for GGEMS simulation
    """
    def __init__(self):
        ggems_lib.create_ggems_voxelized_navigator.restype = ctypes.c_void_p

        ggems_lib.set_phantom_name_ggems_voxelized_navigator.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_phantom_name_ggems_voxelized_navigator.restype = ctypes.c_void_p

        ggems_lib.set_phantom_file_ggems_voxelized_navigator.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_phantom_file_ggems_voxelized_navigator.restype = ctypes.c_void_p

        ggems_lib.set_range_to_material_filename_ggems_voxelized_navigator.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_range_to_material_filename_ggems_voxelized_navigator.restype = ctypes.c_void_p

        ggems_lib.set_geometry_tolerance_ggems_voxelized_navigator.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_geometry_tolerance_ggems_voxelized_navigator.restype = ctypes.c_void_p

        ggems_lib.set_position_ggems_voxelized_navigator.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_position_ggems_voxelized_navigator.restype = ctypes.c_void_p

        self.obj = ggems_lib.create_ggems_voxelized_navigator()

    def set_phantom_name(self, name):
        ggems_lib.set_phantom_name_ggems_voxelized_navigator(self.obj, name.encode('ASCII'))

    def set_phantom_image(self, name):
        ggems_lib.set_phantom_file_ggems_voxelized_navigator(self.obj, name.encode('ASCII'))

    def set_range_to_material(self, name):
        ggems_lib.set_range_to_material_filename_ggems_voxelized_navigator(self.obj, name.encode('ASCII'))

    def set_geometry_tolerance(self, distance, unit):
        ggems_lib.set_geometry_tolerance_ggems_voxelized_navigator(self.obj, distance, unit.encode('ASCII'))

    def set_position(self, offset_x, offset_y, offset_z, unit):
        ggems_lib.set_position_ggems_voxelized_navigator(self.obj, offset_x, offset_y, offset_z, unit.encode('ASCII'))