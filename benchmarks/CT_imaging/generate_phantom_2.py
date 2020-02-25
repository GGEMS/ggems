from ggems import *

# ------------------------------------------------------------------------------
# STEP 1: OpenCL Initialization
opencl_manager.set_context_index(2)  # Activate a context
opencl_manager.print_infos()  # Printing informations about OpenCL

# ------------------------------------------------------------------------------
# STEP 2: Initializing phantom creator manager and setting the informations about the global voxelized volume
phantom_creator_manager.set_dimensions(400, 400, 400)
phantom_creator_manager.set_element_sizes(0.25, 0.25, 0.25, b"mm")
phantom_creator_manager.set_isocenter_positions(-51.5, 0.0, 0.0, b"mm")
phantom_creator_manager.set_output(b"data/phantom_2")
phantom_creator_manager.set_range_output(b"data/range_phantom_2")
phantom_creator_manager.set_material(b"Air")
phantom_creator_manager.initialize()

# ------------------------------------------------------------------------------
# STEP 3: Designing analytical volume(s)
cylinder = GGEMSTube()
cylinder.set_height(100.0, b"mm")
cylinder.set_radius(50.0, b"mm")
cylinder.set_position(0.0, 0.0, 0.0, b"mm")
cylinder.set_label_value(1.0)
cylinder.set_material(b"Water")
cylinder.initialize()
cylinder.draw()
cylinder.delete()

cylinder = GGEMSTube()
cylinder.set_height(100.0, b"mm")
cylinder.set_radius(10.0, b"mm")
cylinder.set_position(0.0, 0.0, 0.0, b"mm")
cylinder.set_label_value(2.0)
cylinder.set_material(b"Uranium")
cylinder.initialize()
cylinder.draw()
cylinder.delete()

# ------------------------------------------------------------------------------
# STEP 4: Saving the final volume
phantom_creator_manager.write()

# ------------------------------------------------------------------------------
# STEP 5: Exit GGEMS safely
opencl_manager.clean()
exit()
