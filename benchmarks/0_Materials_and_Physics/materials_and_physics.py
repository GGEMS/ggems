################################################################################
# Benchmark 0: Materials and Physics                                           #
# In this benchmark, a list of different materials are set to GGEMS, and       #
# different physical effets are used. This benchmark shows how to set a        #
# material in GGEMS and how to call a physical effect applied to this material.#
# 3 materials are set:
#     - Uranium
#     - Water
#     - RibBone
# Then ...
################################################################################

from ggems import *

# Sequence of materials
material_list = ('Uranium', 'Water', 'RibBone')

# Sequence of physical effects
process_list = ('Compton',)

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

# ------------------------------------------------------------------------------
# STEP 4: Add physical processes and initialize them
cross_sections = GGEMSCrossSections()
for process_id in process_list:
  cross_sections.add_process(process_id, 'gamma')

# Intialize cross section tables with previous materials
cross_sections.initialize(materials)

# Printing cross section value from 10 keV to 200 keV
for i in range(10, 200):
  cs = cross_sections.get_cs('Compton', 'Uranium', i, 'keV')
  print (cs, ' cm2.g-1')

# ------------------------------------------------------------------------------
# STEP 5: Exit safely
opencl_manager.clean()
exit()
