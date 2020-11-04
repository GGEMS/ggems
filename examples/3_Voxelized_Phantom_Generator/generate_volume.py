# ************************************************************************
# * This file is part of GGEMS.                                          *
# *                                                                      *
# * GGEMS is free software: you can redistribute it and/or modify        *
# * it under the terms of the GNU General Public License as published by *
# * the Free Software Foundation, either version 3 of the License, or    *
# * (at your option) any later version.                                  *
# *                                                                      *
# * GGEMS is distributed in the hope that it will be useful,             *
# * but WITHOUT ANY WARRANTY; without even the implied warranty of       *
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        *
# * GNU General Public License for more details.                         *
# *                                                                      *
# * You should have received a copy of the GNU General Public License    *
# * along with GGEMS.  If not, see <https://www.gnu.org/licenses/>.      *
# *                                                                      *
# ************************************************************************

from ggems import *

# ------------------------------------------------------------------------------
# STEP 0: Level of verbosity during computation
GGEMSVerbosity(0)

# ------------------------------------------------------------------------------
# STEP 1: OpenCL Initialization
opencl_manager.set_context_index(0)

# ------------------------------------------------------------------------------
# STEP 2: Initializing volume creator manager and setting the informations about the global voxelized volume
volume_creator_manager.set_dimensions(450, 450, 450)
volume_creator_manager.set_element_sizes(0.5, 0.5, 0.5, "mm")
volume_creator_manager.set_output('data/volume')
volume_creator_manager.set_range_output('data/range_volume')
volume_creator_manager.set_material('Air')
volume_creator_manager.set_data_type('MET_INT')
volume_creator_manager.initialize()

# ------------------------------------------------------------------------------
# STEP 3: Designing volume(s)
# Creating a box
box = GGEMSBox(24.0, 36.0, 56.0, 'mm')
box.set_position(-70.0, -30.0, 10.0, 'mm')
box.set_label_value(1)
box.set_material('Water')
box.initialize()
box.draw()
box.delete()

# Creating a tube
tube = GGEMSTube(13.0, 8.0, 50.0, 'mm')
tube.set_position(20.0, 10.0, -2.0, 'mm')
tube.set_label_value(2)
tube.set_material('Calcium')
tube.initialize()
tube.draw()
tube.delete()

# Creating a sphere
sphere = GGEMSSphere(14.0, 'mm')
sphere.set_position(30.0, -30.0, 8.0, 'mm')
sphere.set_label_value(3)
sphere.set_material('Lung')
sphere.initialize()
sphere.draw()
sphere.delete()

# ------------------------------------------------------------------------------
# STEP 4: Saving the final volume
volume_creator_manager.write()

# ------------------------------------------------------------------------------
# STEP 5: Exit GGEMS safely
opencl_manager.clean()
exit()
