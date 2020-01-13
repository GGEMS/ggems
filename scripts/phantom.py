import ggems

# ------------------------------------------------------------------------------
# STEP 1: Setting global verbosity
ggems.GGEMSVerbosity(3)

# ------------------------------------------------------------------------------
# STEP 2: OpenCL Initialization
opencl_manager = ggems.GGEMSOpenCLManager()
opencl_manager.set_context_index(2)  # Activate a context
opencl_manager.print_infos()  # Printing informations about OpenCL

# ------------------------------------------------------------------------------
# STEP 3: Initializing phantom creator manager and setting the informations
# about the voxelized volume
phantom_creator_manager = ggems.GGEMSPhantomCreatorManager()
phantom_creator_manager.set_dimensions(100, 100, 100)
phantom_creator_manager.set_element_sizes(1.5, 1.5, 1.5)
phantom_creator_manager.set_mhd_output(b"phantom1")

# ------------------------------------------------------------------------------
# STEP 4: Designing analytical volume(s)
cylinder = ggems.GGEMSTube()
cylinder.set_height(80.0)
cylinder.set_radius(20.0)
cylinder.set_label_value(10)
cylinder.set_position(0.0, 0.0, 0.0)
cylinder.initialize()
cylinder.draw()
cylinder.delete()
