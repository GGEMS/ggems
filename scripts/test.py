import ggems

# ------------------------------------------------------------------------------
# Setting the verbosity
ggems.Verbosity(1)

# ------------------------------------------------------------------------------
# STEP 1: Initializing OpenCL
opencl_manager = ggems.OpenCLManager()

# Activate a context
opencl_manager.set_context_index(0)

# Printing informations about OpenCL
opencl_manager.print_info()
opencl_manager.print_device()
opencl_manager.print_build_options()
opencl_manager.print_context()
opencl_manager.print_command_queue()
opencl_manager.print_activated_context()

# ------------------------------------------------------------------------------
# STEP 2: Initializing a source


# ------------------------------------------------------------------------------
# STEP 3: GGEMS simulation parameters
ggems_manager = ggems.GGEMSManager()
ggems_manager.set_seed(777)
ggems_manager.set_number_of_particles(861635)
ggems_manager.set_number_of_batchs(1)

# Cross section parameters
ggems_manager.set_cross_section_table_number_of_bins(220)
ggems_manager.set_cross_section_table_energy_min(0.00099)  # in MeV
ggems_manager.set_cross_section_table_energy_max(250.0)  # in MeV

# Add processes and cut for photon
ggems_manager.set_process(b"Compton")
ggems_manager.set_process(b"PhotoElectric")
ggems_manager.set_process(b"Rayleigh")
ggems_manager.set_particle_cut(b"Photon", 0.5)  # 0.5 mm

# Set the geometry tolerance in the range [1mm;1nm]
ggems_manager.set_geometry_tolerance(0.001)  # 1 um

# Initializing the GGEMS simulation
ggems_manager.initialize_simulation()

# Printing RAM status after all initializations
opencl_manager.print_RAM()
