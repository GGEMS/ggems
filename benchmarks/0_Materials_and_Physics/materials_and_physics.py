################################################################################
# Benchmark 0: Materials and Physics                                           #
# In this benchmark, a list of different materials are set to GGEMS, and       #
# different physical effets are used. This benchmark shows how to set a        #
# material in GGEMS and how to call physical effects.                          #
# 3 materials:                                                                 #
#     - Uranium                                                                #
#     - Water                                                                  #
#     - RibBone                                                                #
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ggems import *

# Sequence of materials
material_list = ('Uranium', 'Water', 'RibBone')

# Sequence of physical effects
process_list = ('Compton', 'Photoelectric', 'Rayleigh')

# ------------------------------------------------------------------------------
# STEP 1: Choosing an OpenCL context
opencl_manager.set_context_index(0)

# ------------------------------------------------------------------------------
# STEP 2: Setting GGEMS materials
materials_manager.set_materials('data/materials.txt')

# ------------------------------------------------------------------------------
# STEP 3: Add materials and initialize them
materials = GGEMSMaterials()
for mat_id in material_list:
  materials.add_material(mat_id)

# Initializing materials, and compute some parameters for each material
materials.initialize()
# Priting all informations (by default distance cuts are 1 um)
# materials.print_material_properties()

# Get some useful commands to get infos about material
for mat_id in material_list:
  density = materials.get_density(mat_id)
  photon_energy_cut = materials.get_energy_cut(mat_id, 'gamma', 1.0, 'mm')
  electron_energy_cut = materials.get_energy_cut(mat_id, 'e-', 1.0, 'mm')
  positron_energy_cut = materials.get_energy_cut(mat_id, 'e+', 1.0, 'mm')
  atomic_number_density = materials.get_atomic_number_density(mat_id)
  print('Material:', mat_id)
  print('    Density:', density, ' g.cm-3')
  print('    Photon energy cut (for 2 mm distance):', photon_energy_cut, 'keV')
  print('    Electron energy cut (for 2 mm distance):', electron_energy_cut, 'keV')
  print('    Positron energy cut (for 2 mm distance):', positron_energy_cut, 'keV')
  print('    Atomic number density:', atomic_number_density, 'atoms.cm-3')

#-------------------------------------------------------------------------------
# STEP 4: Defining global parameters for cross-section building
processes_manager.set_cross_section_table_number_of_bins(1024) # Not exceed 1024 bins
processes_manager.set_cross_section_table_energy_min(1.0, 'keV')
processes_manager.set_cross_section_table_energy_max(1.0, 'MeV')

# ------------------------------------------------------------------------------
# STEP 4: Add physical processes and initialize them
cross_sections = GGEMSCrossSections()
for process_id in process_list:
  cross_sections.add_process(process_id, 'gamma')

# Intialize cross section tables with previous materials
cross_sections.initialize(materials)

# Defining 1D X-axis buffer (energy in keV, from 10 to 800 keV, and step of 1 keV)
x = np.arange(10.0, 800.0, 1.0)

# Defining 3D Y-axis buffer (process, material, cross-section in cm2.g-1)
nb_processes = len(process_list)
nb_materials = len(material_list)
nb_cs = len(x)

y = np.zeros((nb_processes, nb_materials, nb_cs))
total = np.zeros((nb_materials, nb_cs))

# Computing cross section values for each physical processes and materials
for p in range(nb_processes):
  for m in range(nb_materials):
    for i in range(nb_cs):
      y[p][m][i] = cross_sections.get_cs(process_list[p], material_list[m], x[i], 'keV')
      total[m][i] = total[m][i] + y[p][m][i]

# Plots
plt.style.use('dark_background')
linestyles = ['--', '-.', ':']

for m in range(nb_materials):
  fig, axis = plt.subplots(1, 1)

  for p in range(nb_processes):
    axis.plot(x, y[p][m], linestyles[p], label='%s' % str(process_list[p]))

  axis.plot(x, total[m], '-', label='Total')

  axis.set_title('Cross-sections %s' % str(material_list[m]))

  axis.set_xscale("log")
  axis.set_yscale("log")

  axis.set_xlabel('Photon energy (keV)')
  axis.set(xlim=(x[0], x[len(x)-1]))
  axis.set_ylabel('(cm2.g-1)')

  axis.legend()
  axis.set_axisbelow(True)
  axis.minorticks_on()
  axis.grid(which='major', linestyle='-', linewidth='0.3')
  axis.grid(which='minor', linestyle=':', linewidth='0.3')

  legend = axis.legend(loc='best', shadow=True, fontsize='large')

  fig.tight_layout()
  plt.savefig('cross_section_%s.png' % str(material_list[m]), bbox_inches='tight', dpi=600)
  plt.close()

# ------------------------------------------------------------------------------
# STEP 5: Exit safely
opencl_manager.clean()
exit()
