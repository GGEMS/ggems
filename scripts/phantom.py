import ggems

# ------------------------------------------------------------------------------
# STEP 1: Setting global verbosity
ggems.GGEMSVerbosity(3)

# ------------------------------------------------------------------------------
# STEP 2: Initializing phantom creator manager and setting the informations
# about the voxelized volume
phantom_creator_manager = ggems.GGEMSPhantomCreatorManager()
phantom_creator_manager.set_position(0.0, 0.0, 0.0)
phantom_creator_manager.set_dimensions(100, 100, 100)
phantom_creator_manager.set_element_sizes(0.5, 0.5, 0.5)
phantom_creator_manager.set_mhd_output(b"phantom1")

# ------------------------------------------------------------------------------
# STEP 3: Designing analytical volume(s)
