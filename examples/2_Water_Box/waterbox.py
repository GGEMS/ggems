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
phantom = GGEMSVoxelizedNavigator()
phantom.set_phantom_name('phantom')
phantom.set_phantom_image('data/waterbox.mhd')
phantom.set_range_to_material('data/range_waterbox.txt')
phantom.set_position(0.0, 0.0, 0.0, 'mm')

# ------------------------------------------------------------------------------
# STEP 5: Physics
processes_manager.add_process('Compton', 'gamma', 'all')
#processes_manager.add_process('Photoelectric', 'gamma', 'all')
#processes_manager.add_process('Rayleigh', 'gamma', 'all')

# Optional options, the following are by default
processes_manager.set_cross_section_table_number_of_bins(220)
processes_manager.set_cross_section_table_energy_min(0.990, 'keV')
processes_manager.set_cross_section_table_energy_max(10.0, 'MeV')

# ------------------------------------------------------------------------------
# STEP 6: Cuts, by default but are 1 um
range_cuts_manager.set_cut('gamma', 1.0, 'mm', 'all')

# ------------------------------------------------------------------------------
# STEP 7: Source
point_source = GGEMSXRaySource()
point_source.set_source_name('point_source')
point_source.set_source_particle_type('gamma')
point_source.set_number_of_particles(1)
point_source.set_position(-595.0, 0.0, 0.0, 'mm')
point_source.set_rotation(0.0, 0.0, 0.0, 'deg')
point_source.set_beam_aperture(0.0, 'deg')
point_source.set_focal_spot_size(0.0, 0.0, 0.0, 'mm')
point_source.set_monoenergy(80.0, 'keV')

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
ggems_manager.tracking_verbose(True)
# ggems_manager.detector_verbose(true/false)

# Initializing the GGEMS simulation
ggems_manager.initialize()

# Start GGEMS simulation
ggems_manager.run()

# ------------------------------------------------------------------------------
# STEP 10: Exit GGEMS safely
opencl_manager.clean()
exit()
