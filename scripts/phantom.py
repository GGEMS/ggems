import ggems

# ------------------------------------------------------------------------------
# STEP 1: Setting global verbosity
ggems.GGEMSVerbosity(0)

# ------------------------------------------------------------------------------
# STEP 2: OpenCL Initialization
opencl_manager = ggems.GGEMSOpenCLManager()
opencl_manager.set_context_index(2)  # Activate a context
opencl_manager.print_infos()  # Printing informations about OpenCL

# ------------------------------------------------------------------------------
# STEP 3: Initializing phantom creator manager and setting the informations
# about the voxelized volume
phantom_creator_manager = ggems.GGEMSPhantomCreatorManager()
phantom_creator_manager.set_dimensions(400, 320, 800)
phantom_creator_manager.set_element_sizes(0.375, 0.5, 0.625)
phantom_creator_manager.set_isocenter_positions(0.0, 0.0, 0.0)
phantom_creator_manager.set_output(b"phantom1", b"mhd")
phantom_creator_manager.initialize()

# ------------------------------------------------------------------------------
# STEP 4: Designing analytical volume(s)
cylinder = ggems.GGEMSTube()
cylinder.set_height(120.0)
cylinder.set_radius(20.0)
cylinder.set_label_value(10)
cylinder.set_position(-20.0, 40.0, 42.0)
cylinder.initialize()
cylinder.draw()
cylinder.delete()

# ------------------------------------------------------------------------------
# STEP 4: Saving the final volume
phantom_creator_manager.write()
