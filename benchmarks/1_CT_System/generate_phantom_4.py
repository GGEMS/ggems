from ggems import *

# ------------------------------------------------------------------------------
# STEP 0: Level of verbosity during computation
GGEMSVerbosity(0)

# ------------------------------------------------------------------------------
# STEP 1: OpenCL Initialization
opencl_manager.set_context_index(0)  # Activate a context

# ------------------------------------------------------------------------------
# STEP 2: Initializing phantom creator manager and setting the informations about the global voxelized volume
volume_creator_manager.set_dimensions(100, 100, 100)
volume_creator_manager.set_element_sizes(0.5, 0.5, 0.5, 'mm')
volume_creator_manager.set_output('data/phantom_4')
volume_creator_manager.set_range_output('data/range_phantom_4')
volume_creator_manager.set_material('Air')
volume_creator_manager.set_data_type('MET_FLOAT')
volume_creator_manager.initialize()

# ------------------------------------------------------------------------------
# STEP 3: Designing analytical volume(s)
cylinder = GGEMSTube()
cylinder.set_height(50.0, 'mm')
cylinder.set_radius(20.0, 'mm')
cylinder.set_position(0.0, 0.0, 0.0, 'mm')
cylinder.set_label_value(1)
cylinder.set_material('Water')
cylinder.initialize()
cylinder.draw()
cylinder.delete()

cylinder = GGEMSTube()
cylinder.set_height(50.0, 'mm')
cylinder.set_radius(5.0, 'mm')
cylinder.set_position(-10.0, 0.0, 0.0, 'mm')
cylinder.set_label_value(2)
cylinder.set_material('Blood')
cylinder.initialize()
cylinder.draw()
cylinder.delete()

cylinder = GGEMSTube()
cylinder.set_height(50.0, 'mm')
cylinder.set_radius(5.0, 'mm')
cylinder.set_position(10.0, 0.0, 0.0, 'mm')
cylinder.set_label_value(3)
cylinder.set_material('Uranium')
cylinder.initialize()
cylinder.draw()
cylinder.delete()

# ------------------------------------------------------------------------------
# STEP 4: Saving the final volume
volume_creator_manager.write()

# ------------------------------------------------------------------------------
# STEP 5: Exit GGEMS safely
opencl_manager.clean()
exit()
