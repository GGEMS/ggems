from ggems import *

# ------------------------------------------------------------------------------
# STEP 1: Choosing an OpenCL context
opencl_manager.set_context_index(2)

# ------------------------------------------------------------------------------
# STEP 2: Visualization


# ------------------------------------------------------------------------------
# STEP 3: Setting GGEMS materials
materials_manager.set_materials(b"data/materials.txt")

# ------------------------------------------------------------------------------
# STEP 4: Phantoms, navigators and systems
phantom_1 = GGEMSVoxelizedPhantomNavigatorImagery()
phantom_1.set_phantom_name(b"phantom_1")
phantom_1.set_phantom_image(b"data/phantom_1.mhd")
phantom_1.set_range_to_material(b"data/range_phantom_1.txt")
phantom_1.set_offset(50.0, 25.0, 0.0, b"mm")

phantom_2 = GGEMSVoxelizedPhantomNavigatorImagery()
phantom_2.set_phantom_name(b"phantom_2")
phantom_2.set_phantom_image(b"data/phantom_2.mhd")
phantom_2.set_range_to_material(b"data/range_phantom_2.txt")
phantom_2.set_offset(0.0, 25.0, 0.0, b"mm")

phantom_3 = GGEMSVoxelizedPhantomNavigatorImagery()
phantom_3.set_phantom_name(b"phantom_3")
phantom_3.set_phantom_image(b"data/phantom_3.mhd")
phantom_3.set_range_to_material(b"data/range_phantom_3.txt")
phantom_3.set_offset(50.0, 25.0, 50.0, b"mm")

phantom_4 = GGEMSVoxelizedPhantomNavigatorImagery()
phantom_4.set_phantom_name(b"phantom_4")
phantom_4.set_phantom_image(b"data/phantom_4.mhd")
phantom_4.set_range_to_material(b"data/range_phantom_4.txt")
phantom_4.set_offset(0.0, 25.0, 50.0, b"mm")

# ------------------------------------------------------------------------------
# STEP 5: Physics
# processes_manager.add_process(b"Compton", b"gamma", b"all")
# or
# processes_manager.add_process(b"Compton", b"gamma", b"phantom_1")
# processes_manager.add_process(b"Compton", b"gamma", b"phantom_2")
# processes_manager.add_process(b"Compton", b"gamma", b"phantom_3")
# processes_manager.add_process(b"Compton", b"gamma", b"phantom_4")

# processes_manager.add_process(b"Photoelectric", b"gamma", b"all")
# or
# processes_manager.add_process(b"Photoelectric", b"gamma", b"phantom_1")
# processes_manager.add_process(b"Photoelectric", b"gamma", b"phantom_2")
# processes_manager.add_process(b"Photoelectric", b"gamma", b"phantom_3")
# processes_manager.add_process(b"Photoelectric", b"gamma", b"phantom_4")

# processes_manager.add_process(b"Rayleigh", b"gamma", b"all")
# or
# processes_manager.add_process(b"Rayleigh", b"gamma", b"phantom_1")
# processes_manager.add_process(b"Rayleigh", b"gamma", b"phantom_2")
# processes_manager.add_process(b"Rayleigh", b"gamma", b"phantom_3")
# processes_manager.add_process(b"Rayleigh", b"gamma", b"phantom_4")

processes_manager.set_cross_section_table_number_of_bins(220)
processes_manager.set_cross_section_table_energy_min(0.99, b"keV")
processes_manager.set_cross_section_table_energy_max(250.0, b"MeV")

# ------------------------------------------------------------------------------
# STEP 6: Cuts
# range_cuts_manager.set_cut("all", "gamma", 1.0, b"mm")
# or
# range_cuts_manager.set_cut("phantom_1", "gamma", 1.0, b"mm")
# range_cuts_manager.set_cut("phantom_2", "gamma", 0.2, b"cm")
# range_cuts_manager.set_cut("phantom_3", "gamma", 13.0, b"um")
# range_cuts_manager.set_cut("phantom_4", "gamma", 0.015, b"m")

# range_cuts_manager.print_infos()

# ------------------------------------------------------------------------------
# STEP 7: Sources
# First source
xray_source_1 = GGEMSXRaySource()
xray_source_1.set_source_name(b"xray_source_1")
xray_source_1.set_source_particle_type(b"photon")
xray_source_1.set_number_of_particles(8616350000)
xray_source_1.set_position(-1000.0, 0.0, 0.0, b"mm")
xray_source_1.set_rotation(0.0, 0.0, 0.0, b"deg")
xray_source_1.set_beam_aperture(5.0, b"deg")
xray_source_1.set_focal_spot_size(0.6, 1.2, 0.0, b"mm")
xray_source_1.set_polyenergy(b"data/spectrum_120kVp_2mmAl.dat")

# Second source
xray_source_2 = GGEMSXRaySource()
xray_source_2.set_source_name(b"xray_source_2")
xray_source_2.set_source_particle_type(b"photon")
xray_source_2.set_number_of_particles(861635)
xray_source_2.set_position(0.0, -1000.0, 0.0, b"mm")
xray_source_2.set_rotation(0.0, 0.0, 0.0, b"deg")
xray_source_2.set_beam_aperture(7.0, b"deg")
xray_source_2.set_focal_spot_size(0.3, 0.5, 0.0, b"mm")
xray_source_2.set_monoenergy(60.2, b"keV")

# ------------------------------------------------------------------------------
# STEP 8: Detector/Digitizer Declaration


# ------------------------------------------------------------------------------
# STEP 9: GGEMS simulation parameters
ggems_manager.set_seed(777) # Optional, if not set, the seed is automatically computed

ggems_manager.opencl_verbose(True)
ggems_manager.material_verbose(True)
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
