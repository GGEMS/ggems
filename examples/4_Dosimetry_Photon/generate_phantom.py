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

import argparse
from ggems import *

# ------------------------------------------------------------------------------
# Read arguments
parser = argparse.ArgumentParser()

parser.add_argument('-d', '--device', required=False, type=int, default=0, help="OpenCL device id")

args = parser.parse_args()

# Get argument
device_id = args.device

# ------------------------------------------------------------------------------
# STEP 0: Level of verbosity during computation
GGEMSVerbosity(0)

# ------------------------------------------------------------------------------
# STEP 1: OpenCL Initialization
opencl_manager.set_device_index(device_id)

# ------------------------------------------------------------------------------
# STEP 2: Initializing volume creator manager and setting the informations about the global voxelized volume
volume_creator_manager.set_dimensions(240, 240, 640)
volume_creator_manager.set_element_sizes(0.5, 0.5, 0.5, 'mm')
volume_creator_manager.set_output('data/phantom')
volume_creator_manager.set_range_output('data/range_phantom')
volume_creator_manager.set_material('Air')
volume_creator_manager.set_data_type('MET_INT')
volume_creator_manager.initialize()

# ------------------------------------------------------------------------------
# STEP 3: Designing analytical volume(s)
phantom = GGEMSTube(50.0, 50.0, 300.0, 'mm')
phantom.set_position(0.0, 0.0, 0.0, 'mm')
phantom.set_label_value(1)
phantom.set_material('Water')
phantom.initialize()
phantom.draw()
phantom.delete()

# ------------------------------------------------------------------------------
# STEP 4: Saving the final volume
volume_creator_manager.write()

# ------------------------------------------------------------------------------
# STEP 5: Exit code
opencl_manager.clean()
exit()
