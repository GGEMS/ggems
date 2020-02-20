from ggems import *

# ------------------------------------------------------------------------------
# STEP 1: Choosing an OpenCL context
opencl_manager.set_context_index(2)
opencl_manager.print_infos()

# ------------------------------------------------------------------------------
# STEP 2: Visualization


# ------------------------------------------------------------------------------
# STEP 3: Setting GGEMS materials
#material_manager.set_materials(b"data/materials.dat")

#material_manager.print_available_chemical_elements()
#material_manager.print_available_materials()

# ------------------------------------------------------------------------------
# STEP 4: Phantom, Navigator and System
# First phantom
#phantom_1 = GGEMSVoxelizedPhantomNavigatorImagery()
#phantom_1.set_phantom_name(b"phantom_1")
#phantom_1.set_phantom_image(b"data/phantom_1.mhd")
#phantom_1.set_range_to_material(b"data/range_phantom_1.dat")
#phantom_1.set_geometry_tolerance(1.0, b"mm")

# Second phantom
#phantom_2 = GGEMSVoxelizedPhantomNavigatorImagery()
#phantom_2.set_phantom_name(b"phantom_2")
#phantom_2.set_phantom_image(b"data/phantom_2.mhd")
#phantom_2.set_range_to_material(b"data/range_phantom_2.dat")
#phantom_2.set_geometry_tolerance(0.1, b"mm")

#phantom_manager.print_infos()

# ------------------------------------------------------------------------------
# STEP 5: Physics Declaration
# First phantom
# processes = ggems.GGEMSProcessesManager()
# processes.add_process("Compton", b"phantom_1")
# processes.add_process("Rayleigh", b"phantom_1")
# processes.add_process("Photoelectric", b"phantom_1")

# processes.print_available_process()
# processes.print_infos()

# processes.set_particle_cut(b"photon", b"phantom_1", 0.5)  # in mm
# set_particle_cut(b"photon", b"phantom_2", 1.5)  # in mm

# physics.set_cross_section_table_number_of_bins(b"phantom_1", 220)
# ggems_manager.set_cross_section_table_energy_min(0.00099)  # in MeV
# ggems_manager.set_cross_section_table_energy_max(250.0)  # in MeV

# ------------------------------------------------------------------------------
# STEP 6: Source Declaration AJOUTER LES UNITES
# First source
#xray_source_1 = GGEMSXRaySource()
#xray_source_1.set_source_name(b"xray_source_1")
#xray_source_1.set_source_particle_type(b"photon")
#xray_source_1.set_number_of_particles(8616350000)
#xray_source_1.set_position(-1000.0, 0.0, 0.0, b"mm")
#xray_source_1.set_rotation(0.0, 0.0, 0.0, b"deg")
#xray_source_1.set_beam_aperture(5.0, b"deg")
#xray_source_1.set_focal_spot_size(0.6, 1.2, 0.0, b"mm")
#xray_source_1.set_polyenergy(b"data/spectrum_120kVp_2mmAl.dat")

# Second source
#xray_source_2 = GGEMSXRaySource()
#xray_source_2.set_source_name(b"xray_source_2")
#xray_source_2.set_source_particle_type(b"photon")
#xray_source_2.set_number_of_particles(861635)
#xray_source_2.set_position(0.0, -1000.0, 0.0, b"mm")
#xray_source_2.set_rotation(0.0, 0.0, 0.0, b"deg")
#xray_source_2.set_beam_aperture(7.0, b"deg")
#xray_source_2.set_focal_spot_size(0.3, 0.5, 0.0, b"mm")
#xray_source_2.set_monoenergy(60.2, b"keV")

#source_manager.print_infos()

# ------------------------------------------------------------------------------
# STEP 7: Detector/Digitizer Declaration


# ------------------------------------------------------------------------------
# STEP X: GGEMS simulation parameters
#ggems_manager.set_seed(777)

# ggems.opencl_verbose(true/false)
# ggems.phantoms_verbose(true/false)
# ggems.source_verbose(true/false)
# ggems.materials_verbose(true/false)
# ggems.processes_verbose(true/false)
# ggems.memory_verbose(true/false)
# ggems.detector_verbose(true/false)
# ggems.tracking_verbose(true/false)

# Initializing the GGEMS simulation
#ggems_manager.initialize()

# Start GGEMS simulation
# ggems.run()
