from ggems_lib import *

class GGEMSPhantomNavigatorManager(object):
    """Class managing phantom navigator in GGEMS
    """
    def __init__(self):
        ggems_lib.get_instance_ggems_phantom_navigator_manager.restype = ctypes.c_void_p

        ggems_lib.print_infos_ggems_phantom_navigator_manager.argtypes = [ctypes.c_void_p]
        ggems_lib.print_infos_ggems_phantom_navigator_manager.restype = ctypes.c_void_p

        self.obj = ggems_lib.get_instance_ggems_phantom_navigator_manager()

    def print_infos(self):
        ggems_lib.print_infos_ggems_phantom_navigator_manager(self.obj)


class GGEMSVoxelizedPhantomNavigatorImagery(object):
    """Class for the voxelized phantom navigator for imagery application
    """
    def __init__(self):
        ggems_lib.create_ggems_voxelized_phantom_navigator_imagery.restype = ctypes.c_void_p

        ggems_lib.set_phantom_name_ggems_voxelized_phantom_navigator_imagery.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_phantom_name_ggems_voxelized_phantom_navigator_imagery.restype = ctypes.c_void_p

        ggems_lib.set_phantom_file_ggems_voxelized_phantom_navigator_imagery.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_phantom_file_ggems_voxelized_phantom_navigator_imagery.restype = ctypes.c_void_p

        ggems_lib.set_range_to_material_filename_ggems_voxelized_phantom_navigator_imagery.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_range_to_material_filename_ggems_voxelized_phantom_navigator_imagery.restype = ctypes.c_void_p

        ggems_lib.set_geometry_tolerance_ggems_voxelized_phantom_navigator_imagery.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_geometry_tolerance_ggems_voxelized_phantom_navigator_imagery.restype = ctypes.c_void_p

        ggems_lib.set_offset_ggems_voxelized_phantom_navigator_imagery.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_offset_ggems_voxelized_phantom_navigator_imagery.restype = ctypes.c_void_p

        self.obj = ggems_lib.create_ggems_voxelized_phantom_navigator_imagery()

    def set_phantom_name(self, name):
        ggems_lib.set_phantom_name_ggems_voxelized_phantom_navigator_imagery(self.obj, name.encode('ASCII'))

    def set_phantom_image(self, name):
        ggems_lib.set_phantom_file_ggems_voxelized_phantom_navigator_imagery(self.obj, name.encode('ASCII'))

    def set_range_to_material(self, name):
        ggems_lib.set_range_to_material_filename_ggems_voxelized_phantom_navigator_imagery(self.obj, name.encode('ASCII'))

    def set_geometry_tolerance(self, distance, unit):
        ggems_lib.set_geometry_tolerance_ggems_voxelized_phantom_navigator_imagery(self.obj, distance, unit.encode('ASCII'))

    def set_offset(self, offset_x, offset_y, offset_z, unit):
        ggems_lib.set_offset_ggems_voxelized_phantom_navigator_imagery(self.obj, offset_x, offset_y, offset_z, unit.encode('ASCII'))