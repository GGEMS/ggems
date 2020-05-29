from ggems import *

# ------------------------------------------------------------------------------
# STEP 0: Level of verbosity during computation
GGEMSVerbosity(3)

# ------------------------------------------------------------------------------
# STEP 1: Choosing an OpenCL context
opencl_manager.set_context_index(0)

# ------------------------------------------------------------------------------
# STEP 2: Visualization


# ------------------------------------------------------------------------------
# STEP 3: Setting GGEMS materials
materials_database_manager.set_materials('data/materials.txt')

# ------------------------------------------------------------------------------
# STEP 4: Phantoms, navigators and systems
phantom_1 = GGEMSVoxelizedPhantomNavigatorImagery()
phantom_1.set_phantom_name('phantom_1')
phantom_1.set_phantom_image('data/phantom_1.mhd')
phantom_1.set_range_to_material('data/range_phantom_1.txt')
phantom_1.set_offset(50.0, 25.0, 0.0, 'mm')

phantom_2 = GGEMSVoxelizedPhantomNavigatorImagery()
phantom_2.set_phantom_name('phantom_2')
phantom_2.set_phantom_image('data/phantom_2.mhd')
phantom_2.set_range_to_material('data/range_phantom_2.txt')
phantom_2.set_offset(0.0, 25.0, 0.0, 'mm')

phantom_3 = GGEMSVoxelizedPhantomNavigatorImagery()
phantom_3.set_phantom_name('phantom_3')
phantom_3.set_phantom_image('data/phantom_3.mhd')
phantom_3.set_range_to_material('data/range_phantom_3.txt')
phantom_3.set_offset(50.0, 25.0, 50.0, 'mm')

phantom_4 = GGEMSVoxelizedPhantomNavigatorImagery()
phantom_4.set_phantom_name('phantom_4')
phantom_4.set_phantom_image('data/phantom_4.mhd')
phantom_4.set_range_to_material('data/range_phantom_4.txt')
phantom_4.set_offset(0.0, 25.0, 50.0, 'mm')

# ------------------------------------------------------------------------------
# STEP 5: Physics
processes_manager.add_process('Compton', 'gamma', 'all')
processes_manager.add_process('Photoelectric', 'gamma', 'all')
processes_manager.add_process('Rayleigh', 'gamma', 'all')

# Optional options, the following are by default
processes_manager.set_cross_section_table_number_of_bins(220) # Not exceed 1000 bins
processes_manager.set_cross_section_table_energy_min(0.99, 'keV')
processes_manager.set_cross_section_table_energy_max(250.0, 'MeV')

# ------------------------------------------------------------------------------
# STEP 6: Cuts, by default but are 1 um
range_cuts_manager.set_cut('gamma', 1.0, 'mm', 'all')

# ------------------------------------------------------------------------------
# STEP 7: Sources
# First source
xray_source_1 = GGEMSXRaySource()
xray_source_1.set_source_name('xray_source_1')
xray_source_1.set_source_particle_type('gamma')
# xray_source_1.set_number_of_particles(8616350000)
xray_source_1.set_number_of_particles(1)
xray_source_1.set_position(-1000.0, 0.0, 0.0, 'mm')
xray_source_1.set_rotation(0.0, 0.0, 0.0, 'deg')
xray_source_1.set_beam_aperture(1.0, 'deg')
xray_source_1.set_focal_spot_size(0.6, 1.2, 0.0, 'mm')
xray_source_1.set_polyenergy('data/spectrum_120kVp_2mmAl.dat')

# Second source
xray_source_2 = GGEMSXRaySource()
xray_source_2.set_source_name('xray_source_2')
xray_source_2.set_source_particle_type('gamma')
# xray_source_2.set_number_of_particles(861635)
xray_source_2.set_number_of_particles(1)
xray_source_2.set_position(-1000.0, 0.0, 0.0, 'mm')
xray_source_2.set_rotation(0.0, 90.0, 0.0, 'deg')
xray_source_2.set_beam_aperture(1.0, 'deg')
xray_source_2.set_focal_spot_size(0.3, 0.5, 0.0, 'mm')
xray_source_2.set_monoenergy(60.2, 'keV')

# ------------------------------------------------------------------------------
# STEP 8: Detector/Digitizer Declaration


# ------------------------------------------------------------------------------
# STEP 9: GGEMS simulation parameters
ggems_manager.set_seed(777) # Optional, if not set, the seed is automatically computed

#ggems_manager.opencl_verbose(False)
ggems_manager.material_database_verbose(False)
ggems_manager.phantom_verbose(True)
ggems_manager.source_verbose(True)
ggems_manager.memory_verbose(True)
ggems_manager.processes_verbose(True)
ggems_manager.range_cuts_verbose(True)
ggems_manager.random_verbose(True)
# ggems_manager.detector_verbose(true/false)
# ggems_manager.tracking_verbose(true/false)

# Initializing the GGEMS simulation
ggems_manager.initialize()

# Start GGEMS simulation
ggems_manager.run()

# ------------------------------------------------------------------------------
# STEP 10: Exit GGEMS safely
opencl_manager.clean()
exit()
