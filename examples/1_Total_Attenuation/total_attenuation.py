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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ggems import *

# ------------------------------------------------------------------------------
# Read arguments
parser = argparse.ArgumentParser()

parser.add_argument('-c', '--context', required=False, type=int, default=0, help="OpenCL context id")
parser.add_argument('-m', '--material', required=True, type=str, help="Set a material name")

args = parser.parse_args()

# Get arguments
material_name = args.material
context_id = args.context

# Sequence of physical effects in GGEMS for total attenuation
process_list = ('Compton', 'Photoelectric', 'Rayleigh')

# ------------------------------------------------------------------------------
# Level of verbosity during computation
GGEMSVerbosity(0)

# ------------------------------------------------------------------------------
# STEP 1: Choosing an OpenCL context
opencl_manager.set_context_index(context_id)

# ------------------------------------------------------------------------------
# STEP 2: Setting GGEMS materials
materials_database_manager.set_materials('../../data/materials.txt')

# ------------------------------------------------------------------------------
# STEP 3: Add materials and initialize them
materials = GGEMSMaterials()
materials.add_material(material_name)

# Initializing materials, and compute some parameters for each material
materials.initialize()

# Get some useful commands to get infos about material
density = materials.get_density(material_name)
photon_energy_cut = materials.get_energy_cut(material_name, 'gamma', 1.0, 'mm')
electron_energy_cut = materials.get_energy_cut(material_name, 'e-', 1.0, 'mm')
positron_energy_cut = materials.get_energy_cut(material_name, 'e+', 1.0, 'mm')
atomic_number_density = materials.get_atomic_number_density(material_name)
print('Material:', material_name)
print('    Density:', density, ' g.cm-3')
print('    Photon energy cut (for 1 mm distance):', photon_energy_cut, 'keV')
print('    Electron energy cut (for 1 mm distance):', electron_energy_cut, 'keV')
print('    Positron energy cut (for 1 mm distance):', positron_energy_cut, 'keV')
print('    Atomic number density:', atomic_number_density, 'atoms.cm-3')

#-------------------------------------------------------------------------------
# STEP 4: Defining global parameters for cross-section building
processes_manager.set_cross_section_table_number_of_bins(2048)
processes_manager.set_cross_section_table_energy_min(10.0, 'keV')
processes_manager.set_cross_section_table_energy_max(1000.0, 'keV')

# ------------------------------------------------------------------------------
# STEP 5: Add physical processes and initialize them
cross_sections = GGEMSCrossSections()
for process_id in process_list:
  cross_sections.add_process(process_id, 'gamma')

# Intialize cross section tables with previous materials
cross_sections.initialize(materials)

# Defining 1D X-axis buffer (energy in keV, from 10 keV to 1 MeV, and step of 1 keV)
energy = np.arange(10, 1000, 0.1)

# Defining 3D Y-axis buffer (process, material, cross-section in cm2.g-1)
nb_processes = len(process_list)
nb_cs = len(energy)

y = np.zeros((nb_processes, nb_cs))
total = np.zeros(nb_cs)

# Computing cross section values for each physical processes and materials
for p in range(nb_processes):
  for i in range(nb_cs):
    y[p][i] = cross_sections.get_cs(process_list[p], material_name, energy[i], 'keV')
    total[i] = total[i] + y[p][i]

# Plots
plt.style.use('dark_background')
linestyles = ['--', '-.', ':']

fig, axis = plt.subplots(1, 1)

for p in range(nb_processes):
  axis.plot(energy, y[p], linestyles[p], label='{}'.format(process_list[p]))

axis.plot(energy, total, '-', label='Total Attenuation')

axis.set_title('{} Total Attenuation'.format(material_name))

axis.set_xscale("log")
axis.set_yscale("log")

axis.set_xlabel('Photon Energy [keV]')
axis.set(xlim=(energy[0], energy[len(energy)-1]))
axis.set_ylabel('Cross Section [cm2.g-1]')

axis.legend()
axis.set_axisbelow(True)
axis.minorticks_on()
axis.grid(which='major', linestyle='-', linewidth='0.3')
axis.grid(which='minor', linestyle=':', linewidth='0.3')

legend = axis.legend(loc='best', shadow=True, fontsize='large')

fig.tight_layout()
plt.savefig('{}_Total_Attenuation.png'.format(material_name), bbox_inches='tight', dpi=600)
plt.close()

# ------------------------------------------------------------------------------
# STEP 6: Exit safely
opencl_manager.clean()
exit()
