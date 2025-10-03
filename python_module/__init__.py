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
from .ggems_lib import *
from .ggems_opencl import GGEMSOpenCLManager
from .ggems_opengl import GGEMSOpenGLManager
from .ggems_ram import GGEMSRAMManager
from .ggems_materials import GGEMSMaterialsDatabaseManager, GGEMSMaterials
from .ggems_systems import GGEMSCTSystem
from .ggems_phantoms import GGEMSVoxelizedPhantom, GGEMSMeshedPhantom, GGEMSWorld
from .ggems_sources import GGEMSXRaySource, GGEMSSourceManager, GGEMSVoxelizedSource
from .ggems_processes import GGEMSProcessesManager, GGEMSRangeCutsManager, GGEMSCrossSections
from .ggems_volume_creator import GGEMSVolumeCreatorManager, GGEMSTube, GGEMSBox, GGEMSSphere
from .ggems_dosimetry import GGEMSDosimetryCalculator
from .ggems_profiler import GGEMSProfilerManager
from .ggems_attenuation import GGEMSAttenuations

class GGEMS(object):
    """GGEMS class managing the simulation
    """
    def __init__(self):
        ggems_lib.create_ggems.restype = ctypes.c_void_p

        ggems_lib.delete_ggems.argtypes = [ctypes.c_void_p]
        ggems_lib.delete_ggems.restype = ctypes.c_void_p

        ggems_lib.initialize_ggems.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        ggems_lib.initialize_ggems.restype = ctypes.c_void_p

        ggems_lib.set_opencl_verbose_ggems.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.set_opencl_verbose_ggems.restype = ctypes.c_void_p

        ggems_lib.set_material_database_verbose_ggems.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.set_material_database_verbose_ggems.restype = ctypes.c_void_p

        ggems_lib.set_source_ggems.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.set_source_ggems.restype = ctypes.c_void_p

        ggems_lib.set_navigator_ggems.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.set_navigator_ggems.restype = ctypes.c_void_p

        ggems_lib.set_memory_ram_ggems.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.set_memory_ram_ggems.restype = ctypes.c_void_p

        ggems_lib.set_process_ggems.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.set_process_ggems.restype = ctypes.c_void_p

        ggems_lib.set_range_cuts_ggems.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.set_range_cuts_ggems.restype = ctypes.c_void_p

        ggems_lib.set_profiling_ggems.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.set_profiling_ggems.restype = ctypes.c_void_p

        ggems_lib.set_random_ggems.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.set_random_ggems.restype = ctypes.c_void_p

        ggems_lib.set_tracking_ggems.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_int]
        ggems_lib.set_tracking_ggems.restype = ctypes.c_void_p

        ggems_lib.run_ggems.argtypes = [ctypes.c_void_p]
        ggems_lib.run_ggems.restype = ctypes.c_void_p

        self.obj = ggems_lib.create_ggems()

    def delete(self):
        ggems_lib.delete_ggems(self.obj)

    def initialize(self, seed = 0):
        ggems_lib.initialize_ggems(self.obj, seed)

    def run(self):
        ggems_lib.run_ggems(self.obj)

    def opencl_verbose(self, flag):
        ggems_lib.set_opencl_verbose_ggems(self.obj, flag)

    def material_database_verbose(self, flag):
        ggems_lib.set_material_database_verbose_ggems(self.obj, flag)

    def navigator_verbose(self, flag):
        ggems_lib.set_navigator_ggems(self.obj, flag)

    def source_verbose(self, flag):
        ggems_lib.set_source_ggems(self.obj, flag)

    def memory_verbose(self, flag):
        ggems_lib.set_memory_ram_ggems(self.obj, flag)

    def process_verbose(self, flag):
        ggems_lib.set_process_ggems(self.obj, flag)

    def profiling_verbose(self, flag):
        ggems_lib.set_profiling_ggems(self.obj, flag)

    def range_cuts_verbose(self, flag):
        ggems_lib.set_range_cuts_ggems(self.obj, flag)

    def random_verbose(self, flag):
        ggems_lib.set_random_ggems(self.obj, flag)

    def tracking_verbose(self, flag, particle_id):
        ggems_lib.set_tracking_ggems(self.obj, flag, particle_id)


def clean_safely():
    GGEMSOpenCLManager().clean()
    GGEMSVolumeCreatorManager().clean();
    GGEMSSourceManager().clean();
