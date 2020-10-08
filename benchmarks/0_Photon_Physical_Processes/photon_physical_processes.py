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

################################################################################
# Benchmark 0: Photon Physical Processes                                       #
# Physic tables for photon processes are compared between GGEMS and G4 (version#
# 10.6 patch 2). Scattered gamma energies and gamma angles are also evaluated. #
# Electrons are deactivated                                                    #
# NIST values are also plotted for physic tables                               #
#                                                                              #
# Processes and models are:                                                    #
# * Compton Scattering: the model in G4 is G4KleinNishinaCompton (without      #
# atomic shell effect)                                                         #
# * Rayleigh Scattering:                                                       #
# * Photoelectric Effect:                                                      #
#                                                                              #
# Materials used for the benchmark:                                            #
# * Lead                                                                       #
# * Water                                                                      #
# * Calcium                                                                    #
################################################################################

import numpy as np
import math
import matplotlib.pyplot as plt

from ggems import *

# Compared softwares
database_list = ('NIST', 'G4', 'GGEMS')

# Sequence of materials
# material_list = ('Calcium', 'Water', 'Lead')
material_list = ('Water',)

# Sequence of physical effects
# process_list = ('Compton', 'Photoelectric', 'Rayleigh')
process_list = ('Compton',)

# Read data from G4 and NIST files
nist_data_water = np.loadtxt('data/Water_1keV_to_10MeV_NIST.dat')
g4_data_water = np.loadtxt('data/Water_1keV_to_10MeV_Geant4.dat')

# Extract energy (1 keV to 10 MeV)
energy = nist_data_water[:,0]

# Defining 4D buffer (material, process, software and cross-section in cm2.g-1)
nb_materials = len(material_list)
nb_processes = len(process_list)
nb_databases = len(database_list)
nb_bins = len(energy)

# Extract cross section from NIST (col 0) and G4 (col 1)
cross_section_data = np.zeros((nb_materials, nb_processes, nb_databases, nb_bins))
# Water
cross_section_data[0][0][0] = nist_data_water[:,1] # Compton Scattering, NIST
cross_section_data[0][0][1] = g4_data_water[:,1] # Compton Scattering, G4

# ------------------------------------------------------------------------------
# Level of verbosity during computation
GGEMSVerbosity(0)

# ------------------------------------------------------------------------------
# STEP 1: Choosing an OpenCL context
opencl_manager.set_context_index(0)

# ------------------------------------------------------------------------------
# STEP 2: Setting GGEMS materials
materials_database_manager.set_materials('data/materials.txt')

# ------------------------------------------------------------------------------
# STEP 3: Add materials and initialize them
materials = GGEMSMaterials()
for mat_id in material_list:
  materials.add_material(mat_id)

# Initializing materials, and compute some parameters for each material
materials.initialize()

# Get some useful commands to get infos about material
for mat_id in material_list:
  density = materials.get_density(mat_id)
  photon_energy_cut = materials.get_energy_cut(mat_id, 'gamma', 1.0, 'mm')
  electron_energy_cut = materials.get_energy_cut(mat_id, 'e-', 1.0, 'mm')
  positron_energy_cut = materials.get_energy_cut(mat_id, 'e+', 1.0, 'mm')
  atomic_number_density = materials.get_atomic_number_density(mat_id)
  print('Material:', mat_id)
  print('    Density:', density, ' g.cm-3')
  print('    Photon energy cut (for 1 mm distance):', photon_energy_cut, 'keV')
  print('    Electron energy cut (for 1 mm distance):', electron_energy_cut, 'keV')
  print('    Positron energy cut (for 1 mm distance):', positron_energy_cut, 'keV')
  print('    Atomic number density:', atomic_number_density, 'atoms.cm-3')

#-------------------------------------------------------------------------------
# STEP 4: Defining global parameters for cross-section building
processes_manager.set_cross_section_table_number_of_bins(220) # Not exceed 2048 bins
processes_manager.set_cross_section_table_energy_min(1.0, 'keV')
processes_manager.set_cross_section_table_energy_max(10.0, 'MeV')

# ------------------------------------------------------------------------------
# STEP 5: Add physical processes and initialize them
cross_sections = GGEMSCrossSections()
for process_id in process_list:
  cross_sections.add_process(process_id, 'gamma')

# Intialize cross section tables with previous materials
cross_sections.initialize(materials)

# Computing cross sections with GGEMS
for m in range(nb_materials):
  for p in range(nb_processes):
    for e in range(nb_bins):
      cross_section_data[m][p][2][e] = cross_sections.get_cs(process_list[p], material_list[m], energy[e], 'MeV')

# Plots
plt.style.use('dark_background')
linestyles = ['--', '-.', ':']

for m in range(nb_materials):
  for p in range(nb_processes):
    fig, axis = plt.subplots(1, 1)

    for s in range(nb_databases):
      axis.plot(energy, cross_section_data[m][p][s], linestyles[s], label='{}'.format(database_list[s]))

    axis.set_title('{} {} Cross Section'.format(material_list[m], process_list[p]))

    axis.set_xscale("log")
    #axis.set_yscale("log")

    axis.set_xlabel('Photon Energy [MeV]')
    axis.set(xlim=(energy[0], energy[len(energy)-1]))
    axis.set_ylabel('Cross Section [cm2.g-1]')

    axis.legend()
    axis.set_axisbelow(True)
    axis.minorticks_on()
    axis.grid(which='major', linestyle='-', linewidth='0.3')
    axis.grid(which='minor', linestyle=':', linewidth='0.3')

    legend = axis.legend(loc='best', shadow=True, fontsize='large')

    fig.tight_layout()
    plt.savefig('{}_{}_Cross_Section.png'.format(material_list[m], process_list[p]), bbox_inches='tight', dpi=600)
    plt.close()

# Print results (MAE, MSE and RMSE), reference is G4
print('Error evaluation for each process and material. G4 data are considered as the reference values and GGEMS data the predicted values')
for m in range(nb_materials):
  print(material_list[m])
  for p in range(nb_processes):
    print('{} process:'.format(process_list[p]))
    mae = np.subtract(cross_section_data[m][p][1], cross_section_data[m][p][2]).mean()
    mse = np.square(np.subtract(cross_section_data[m][p][1], cross_section_data[m][p][2])).mean()
    rmse = math.sqrt(mse)
    print('    * MAE: ', mae)
    print('    * MSE: ', mse)
    print('    * RMSE: ', rmse)

# ------------------------------------------------------------------------------
# STEP 6: Exit safely
opencl_manager.clean()
exit()
