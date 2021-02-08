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

import argparse
from ggems import *

# ------------------------------------------------------------------------------
# Read arguments
parser = argparse.ArgumentParser()

parser.add_argument('-c', '--context', required=False, type=int, default=0, help="OpenCL context id")
parser.add_argument('-m', '--material', required=True, type=str, help="Set a material name")
parser.add_argument('-p', '--process', required=True, type=str, help="Set a physical process", choices=['Compton', 'Photoelectric', 'Rayleigh'])
parser.add_argument('-e', '--energy', required=True, type=float, help="Set an energy in MeV")

args = parser.parse_args()

# Get arguments
material_name = args.material
energy_MeV = args.energy
process_name = args.process
context_id = args.context

# ------------------------------------------------------------------------------
# Level of verbosity during GGEMS execution
GGEMSVerbosity(1)

# ------------------------------------------------------------------------------
# STEP 1: Choosing an OpenCL context
opencl_manager.set_context_index(context_id)

# ------------------------------------------------------------------------------
# STEP 2: Setting GGEMS materials
materials_database_manager.set_materials('../../data/materials.txt')

# ------------------------------------------------------------------------------
# STEP 3: Add material and initialize it
materials = GGEMSMaterials()
materials.add_material(material_name)
# Initializing materials, and compute some parameters
materials.initialize()

# Printing useful infos
print('Material:', material_name)
print('    Density:', materials.get_density(material_name), ' g.cm-3')
print('    Photon energy cut (for 1 mm distance):', materials.get_energy_cut(material_name, 'gamma', 1.0, 'mm'), 'keV')
print('    Electron energy cut (for 1 mm distance):', materials.get_energy_cut(material_name, 'e-', 1.0, 'mm'), 'keV')
print('    Positron energy cut (for 1 mm distance):', materials.get_energy_cut(material_name, 'e+', 1.0, 'mm'), 'keV')
print('    Atomic number density:', materials.get_atomic_number_density(material_name), 'atoms.cm-3')

#-------------------------------------------------------------------------------
# STEP 4: Defining global parameters for cross-section building
processes_manager.set_cross_section_table_number_of_bins(220) # Not exceed 2048 bins
processes_manager.set_cross_section_table_energy_min(1.0, 'keV')
processes_manager.set_cross_section_table_energy_max(10.0, 'MeV')

# ------------------------------------------------------------------------------
# STEP 5: Add physical process and initialize it
cross_sections = GGEMSCrossSections()
cross_sections.add_process(process_name, 'gamma')
# Intialize cross section tables with previous materials
cross_sections.initialize(materials)

print('At ', energy_MeV, ' MeV, cross section is ', cross_sections.get_cs(process_name, material_name, energy_MeV, 'MeV'), 'cm2.g-1')

# ------------------------------------------------------------------------------
# STEP 6: Exit safely
opencl_manager.clean()
exit()
