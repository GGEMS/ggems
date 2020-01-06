import ggems
import ctypes

# ------------------------------------------------------------------------------
# Setting global verbosity
ggems.GGEMSVerbosity(3)

# ------------------------------------------------------------------------------
# STEP 1: Initializing OpenCL
opencl_manager = ggems.GGEMSOpenCLManager()
opencl_manager.set_context_index(2)  # Activate a context
opencl_manager.print_infos()  # Printing informations about OpenCL

# ------------------------------------------------------------------------------
# STEP 2: Initializing a source
xray_source = ggems.GGEMSXRaySource()
xray_source.set_position(-1000.0, 10.0, 50.0)  # in mm
xray_source.set_source_particle_type(b"photon")
xray_source.set_beam_aperture(5.0)  # in degree
xray_source.set_focal_spot_size(0.6, 1.2, 0.0)  # in mm
xray_source.set_rotation(0.0, 0.0, 0.0)  # in degree
xray_source.set_polyenergy(b"scripts/spectrum_120kVp_2mmAl.dat")
xray_source.print_infos()

# ------------------------------------------------------------------------------
# STEP 3: GGEMS simulation parameters
ggems_manager = ggems.GGEMSManager()
ggems_manager.set_seed(777)
ggems_manager.set_number_of_particles(861635)

# Cross section parameters
ggems_manager.set_cross_section_table_number_of_bins(220)
ggems_manager.set_cross_section_table_energy_min(0.00099)  # in MeV
ggems_manager.set_cross_section_table_energy_max(250.0)  # in MeV

# Add processes and cut for photon
ggems_manager.set_process(b"compton")
ggems_manager.set_process(b"photoElectric")
ggems_manager.set_process(b"rayleigh")
ggems_manager.set_particle_cut(b"photon", 0.5)  # in mm

# Set the geometry tolerance in the range [1mm;1nm]
ggems_manager.set_geometry_tolerance(0.001)  # in mm

# Initializing the GGEMS simulation
ggems_manager.initialize()
# Printing RAM status after all initializations
opencl_manager.print_RAM()

# Start GGEMS simulation
ggems_manager.run()

# Freeing properly the memory
xray_source.delete()
opencl_manager.clean()
