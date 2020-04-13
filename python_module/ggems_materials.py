from ggems_lib import *

class GGEMSMaterialsManager(object):
    """Class handling the materials database in GGEMS
    """
    def __init__(self):
        ggems_lib.get_instance_materials_manager.restype = ctypes.c_void_p

        ggems_lib.set_materials_ggems_materials_manager.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_materials_ggems_materials_manager.restype = ctypes.c_void_p

        ggems_lib.print_available_chemical_elements_ggems_materials_manager.argtypes = [ctypes.c_void_p]
        ggems_lib.print_available_chemical_elements_ggems_materials_manager.restype = ctypes.c_void_p

        ggems_lib.print_available_materials_ggems_materials_manager.argtypes = [ctypes.c_void_p]
        ggems_lib.print_available_materials_ggems_materials_manager.restype = ctypes.c_void_p

        self.obj = ggems_lib.get_instance_materials_manager()

    def set_materials(self, filename):
        ggems_lib.set_materials_ggems_materials_manager(self.obj, filename.encode('ASCII'))

    def print_available_chemical_elements(self):
        ggems_lib.print_available_chemical_elements_ggems_materials_manager(self.obj)

    def print_available_materials(self):
        ggems_lib.print_available_materials_ggems_materials_manager(self.obj)


class GGEMSMaterials(object):
    """ Class handling materials one by one in GGEMS
    """
    def __init__(self):
        ggems_lib.create_ggems_materials.restype = ctypes.c_void_p

        ggems_lib.add_material_ggems_materials.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.add_material_ggems_materials.restype = ctypes.c_void_p

        ggems_lib.initialize_ggems_materials.argtypes = [ctypes.c_void_p]
        ggems_lib.initialize_ggems_materials.restype = ctypes.c_void_p

        ggems_lib.print_material_properties_ggems_materials.argtypes = [ctypes.c_void_p]
        ggems_lib.print_material_properties_ggems_materials.restype = ctypes.c_void_p

        ggems_lib.get_density_ggems_materials.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.get_density_ggems_materials.restype = ctypes.c_float

        ggems_lib.get_energy_cut_ggems_materials.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.get_energy_cut_ggems_materials.restype = ctypes.c_float

        ggems_lib.get_atomic_number_density_ggems_materials.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.get_atomic_number_density_ggems_materials.restype = ctypes.c_float

        self.obj = ggems_lib.create_ggems_materials()

    def add_material(self, material):
        ggems_lib.add_material_ggems_materials(self.obj, material.encode('ASCII'))

    def initialize(self):
        ggems_lib.initialize_ggems_materials(self.obj)

    def print_material_properties(self):
        ggems_lib.print_material_properties_ggems_materials(self.obj)

    def get_density(self, material_name):
        return ggems_lib.get_density_ggems_materials(self.obj, material_name.encode('ASCII'))

    def get_energy_cut(self, material_name, particle_name, value, unit):
        return ggems_lib.get_energy_cut_ggems_materials(self.obj, material_name.encode('ASCII'), particle_name.encode('ASCII'), value, unit.encode('ASCII'))

    def get_atomic_number_density(self, material_name):
        return ggems_lib.get_atomic_number_density_ggems_materials(self.obj, material_name.encode('ASCII'))