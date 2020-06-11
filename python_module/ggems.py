# Import all GGEMS C++ singletons
from ggems_lib import *
from ggems_opencl import GGEMSOpenCLManager
from ggems_materials import GGEMSMaterialsDatabaseManager, GGEMSMaterials
from ggems_navigators import GGEMSVoxelizedNavigator
from ggems_sources import GGEMSSourceManager, GGEMSXRaySource
from ggems_processes import GGEMSProcessesManager, GGEMSRangeCutsManager, GGEMSCrossSections
from ggems_volume_creator import GGEMSVolumeCreatorManager, GGEMSTube


class GGEMSManager(object):
    """GGEMS class managing the simulation
    """
    def __init__(self):
        ggems_lib.get_instance_ggems_manager.restype = ctypes.c_void_p

        ggems_lib.set_seed_ggems_manager.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        ggems_lib.set_seed_ggems_manager.restype = ctypes.c_void_p

        ggems_lib.initialize_ggems_manager.argtypes = [ctypes.c_void_p]
        ggems_lib.initialize_ggems_manager.restype = ctypes.c_void_p

        ggems_lib.set_opencl_verbose_ggems_manager.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.set_opencl_verbose_ggems_manager.restype = ctypes.c_void_p

        ggems_lib.set_material_database_verbose_ggems_manager.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.set_material_database_verbose_ggems_manager.restype = ctypes.c_void_p

        ggems_lib.set_source_ggems_manager.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.set_source_ggems_manager.restype = ctypes.c_void_p

        ggems_lib.set_phantom_ggems_manager.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.set_phantom_ggems_manager.restype = ctypes.c_void_p

        ggems_lib.set_memory_ram_ggems_manager.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.set_memory_ram_ggems_manager.restype = ctypes.c_void_p

        ggems_lib.set_processes_ggems_manager.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.set_processes_ggems_manager.restype = ctypes.c_void_p

        ggems_lib.set_range_cuts_ggems_manager.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.set_range_cuts_ggems_manager.restype = ctypes.c_void_p

        ggems_lib.set_random_ggems_manager.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.set_random_ggems_manager.restype = ctypes.c_void_p

        ggems_lib.run_ggems_manager.argtypes = [ctypes.c_void_p]
        ggems_lib.run_ggems_manager.restype = ctypes.c_void_p

        self.obj = ggems_lib.get_instance_ggems_manager()

    def set_seed(self, seed):
        ggems_lib.set_seed_ggems_manager(self.obj, seed)

    def initialize(self):
        ggems_lib.initialize_ggems_manager(self.obj)

    def run(self):
        ggems_lib.run_ggems_manager(self.obj)

    def opencl_verbose(self, flag):
        ggems_lib.set_opencl_verbose_ggems_manager(self.obj, flag)

    def material_database_verbose(self, flag):
        ggems_lib.set_material_database_verbose_ggems_manager(self.obj, flag)

    def phantom_verbose(self, flag):
        ggems_lib.set_phantom_ggems_manager(self.obj, flag)

    def source_verbose(self, flag):
        ggems_lib.set_source_ggems_manager(self.obj, flag)

    def memory_verbose(self, flag):
        ggems_lib.set_memory_ram_ggems_manager(self.obj, flag)

    def processes_verbose(self, flag):
        ggems_lib.set_processes_ggems_manager(self.obj, flag)

    def range_cuts_verbose(self, flag):
        ggems_lib.set_range_cuts_ggems_manager(self.obj, flag)

    def random_verbose(self, flag):
        ggems_lib.set_random_ggems_manager(self.obj, flag)


# ------------------------------------------------------------------------------
# Calling all C++ singleton managers
opencl_manager = GGEMSOpenCLManager()
materials_database_manager = GGEMSMaterialsDatabaseManager()
source_manager = GGEMSSourceManager()
processes_manager = GGEMSProcessesManager()
range_cuts_manager = GGEMSRangeCutsManager()
ggems_manager = GGEMSManager()
volume_creator_manager = GGEMSVolumeCreatorManager()
