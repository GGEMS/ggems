import ggems

# ------------------------------------------------------------------------------
# STEP 1: Setting global verbosity
ggems.GGEMSVerbosity(3)

# ------------------------------------------------------------------------------
# STEP 2: Initializing phantom creator manager and setting the informations
# about the voxelized volume
phantom_creator_manager = ggems.GGEMSPhantomCreatorManager()

# ------------------------------------------------------------------------------
# STEP 3: Designing analytical volume(s)