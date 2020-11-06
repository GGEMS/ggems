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

class GGEMSCrossSections(object):
    """ Class handling cross sections in GGEMS
    """
    def __init__(self):
        ggems_lib.create_ggems_cross_sections.restype = ctypes.c_void_p

        ggems_lib.add_process_ggems_cross_sections.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        ggems_lib.add_process_ggems_cross_sections.restype = ctypes.c_void_p

        ggems_lib.initialize_ggems_cross_sections.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        ggems_lib.initialize_ggems_cross_sections.restype = ctypes.c_void_p

        ggems_lib.get_cs_cross_sections.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.get_cs_cross_sections.restype = ctypes.c_float

        self.obj = ggems_lib.create_ggems_cross_sections()

    def add_process(self, process_name, particle_name):
        ggems_lib.add_process_ggems_cross_sections(self.obj, process_name.encode('ASCII'), particle_name.encode('ASCII'))

    def initialize(self, material_p):
        ggems_lib.initialize_ggems_cross_sections(self.obj, material_p.obj)

    def get_cs(self, process_name, material_name, energy, unit):
        return ggems_lib.get_cs_cross_sections(self.obj, process_name.encode('ASCII'), material_name.encode('ASCII'), energy, unit.encode('ASCII'))


class GGEMSRangeCutsManager(object):
    """Class managing the range cuts in GGEMS
    """
    def __init__(self):
        ggems_lib.get_instance_range_cuts_manager.restype = ctypes.c_void_p

        ggems_lib.set_cut_range_cuts_manager.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_cut_range_cuts_manager.restype = ctypes.c_void_p

        self.obj = ggems_lib.get_instance_range_cuts_manager()
    
    def set_cut(self, particle, value, unit, phantom = b'all'):
        ggems_lib.set_cut_range_cuts_manager(self.obj, phantom.encode('ASCII'), particle.encode('ASCII'), value, unit.encode('ASCII'))


class GGEMSProcessesManager(object):
    """Class managing the processes in GGEMS
    """
    def __init__(self):
        ggems_lib.get_instance_processes_manager.restype = ctypes.c_void_p

        ggems_lib.set_cross_section_table_number_of_bins_processes_manager.argtypes = [ctypes.c_void_p, ctypes.c_short]
        ggems_lib.set_cross_section_table_number_of_bins_processes_manager.restype = ctypes.c_void_p

        ggems_lib.set_cross_section_table_minimum_energy_processes_manager.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_cross_section_table_minimum_energy_processes_manager.restype = ctypes.c_void_p

        ggems_lib.set_cross_section_table_maximum_energy_processes_manager.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_cross_section_table_maximum_energy_processes_manager.restype = ctypes.c_void_p

        ggems_lib.print_infos_processes_manager.argtypes = [ctypes.c_void_p]
        ggems_lib.print_infos_processes_manager.restype = ctypes.c_void_p

        ggems_lib.print_available_processes_manager.argtypes = [ctypes.c_void_p]
        ggems_lib.print_available_processes_manager.restype = ctypes.c_void_p

        ggems_lib.add_process_processes_manager.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_bool]
        ggems_lib.add_process_processes_manager.restype = ctypes.c_void_p

        ggems_lib.print_tables_processes_manager.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.print_tables_processes_manager.restype = ctypes.c_void_p

        self.obj = ggems_lib.get_instance_processes_manager()

    def set_cross_section_table_number_of_bins(self, number_of_bins):
        ggems_lib.set_cross_section_table_number_of_bins_processes_manager(self.obj, number_of_bins)

    def set_cross_section_table_energy_min(self, energy, unit):
        ggems_lib.set_cross_section_table_minimum_energy_processes_manager(self.obj, energy, unit.encode('ASCII'))

    def set_cross_section_table_energy_max(self, energy, unit):
        ggems_lib.set_cross_section_table_maximum_energy_processes_manager(self.obj, energy, unit.encode('ASCII'))

    def print_available_processes(self):
        ggems_lib.print_available_processes_manager(self.obj)

    def print_infos(self):
        ggems_lib.print_infos_processes_manager(self.obj)

    def add_process(self, process_name, particle_name, phantom_name=b'all', is_secondary=False):
        ggems_lib.add_process_processes_manager(self.obj, process_name.encode('ASCII'), particle_name.encode('ASCII'), phantom_name.encode('ASCII'), is_secondary)

    def print_tables(self, flag):
        ggems_lib.print_tables_processes_manager(self.obj, flag)