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

# Import all GGEMS C++ singletons
from ggems_lib import *
from ggems_opencl import GGEMSOpenCLManager
from ggems_materials import GGEMSMaterialsDatabaseManager, GGEMSMaterials
from ggems_systems import GGEMSCTSystem
from ggems_phantoms import GGEMSVoxelizedPhantom
from ggems_sources import GGEMSXRaySource
from ggems_processes import GGEMSProcessesManager, GGEMSRangeCutsManager, GGEMSCrossSections
from ggems_volume_creator import GGEMSVolumeCreatorManager, GGEMSTube, GGEMSBox, GGEMSSphere

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

        ggems_lib.set_navigator_ggems_manager.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.set_navigator_ggems_manager.restype = ctypes.c_void_p

        ggems_lib.set_memory_ram_ggems_manager.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.set_memory_ram_ggems_manager.restype = ctypes.c_void_p

        ggems_lib.set_process_ggems_manager.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.set_process_ggems_manager.restype = ctypes.c_void_p

        ggems_lib.set_range_cuts_ggems_manager.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.set_range_cuts_ggems_manager.restype = ctypes.c_void_p

        ggems_lib.set_kernel_ggems_manager.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.set_kernel_ggems_manager.restype = ctypes.c_void_p

        ggems_lib.set_random_ggems_manager.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.set_random_ggems_manager.restype = ctypes.c_void_p

        ggems_lib.set_tracking_ggems_manager.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_int]
        ggems_lib.set_tracking_ggems_manager.restype = ctypes.c_void_p

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

    def navigator_verbose(self, flag):
        ggems_lib.set_navigator_ggems_manager(self.obj, flag)

    def source_verbose(self, flag):
        ggems_lib.set_source_ggems_manager(self.obj, flag)

    def memory_verbose(self, flag):
        ggems_lib.set_memory_ram_ggems_manager(self.obj, flag)

    def process_verbose(self, flag):
        ggems_lib.set_process_ggems_manager(self.obj, flag)

    def kernel_verbose(self, flag):
        ggems_lib.set_kernel_ggems_manager(self.obj, flag)

    def range_cuts_verbose(self, flag):
        ggems_lib.set_range_cuts_ggems_manager(self.obj, flag)

    def random_verbose(self, flag):
        ggems_lib.set_random_ggems_manager(self.obj, flag)

    def tracking_verbose(self, flag, particle_id):
        ggems_lib.set_tracking_ggems_manager(self.obj, flag, particle_id)


# ------------------------------------------------------------------------------
# Calling all C++ singleton managers
opencl_manager = GGEMSOpenCLManager()
materials_database_manager = GGEMSMaterialsDatabaseManager()
processes_manager = GGEMSProcessesManager()
range_cuts_manager = GGEMSRangeCutsManager()
ggems_manager = GGEMSManager()
volume_creator_manager = GGEMSVolumeCreatorManager()
