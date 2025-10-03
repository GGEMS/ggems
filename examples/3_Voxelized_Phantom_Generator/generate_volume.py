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
parser = argparse.ArgumentParser(
  prog='generate_volume.py',
  description='-->> 3 - Generate Volume Example <<--',
  epilog='',
  formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('-d', '--device', required=False, type=int, default=0, help="OpenCL device id")
parser.add_argument('-v', '--verbose', required=False, type=int, default=0, help="Set level of verbosity")

args = parser.parse_args()

# Get argument
device_id = args.device
verbosity_level = args.verbose

# ------------------------------------------------------------------------------
# STEP 0: Level of verbosity during computation
GGEMSVerbosity(verbosity_level)

# ------------------------------------------------------------------------------
# STEP 1: Calling C++ singleton
opencl_manager = GGEMSOpenCLManager()
volume_creator_manager = GGEMSVolumeCreatorManager()
profiler_manager = GGEMSProfilerManager()
ram_manager = GGEMSRAMManager()

# ------------------------------------------------------------------------------
# STEP 2: OpenCL Initialization
opencl_manager.set_device_index(device_id)

# ------------------------------------------------------------------------------
# STEP 3: Initializing volume creator manager and setting the informations about the global voxelized volume
volume_creator_manager.set_dimensions(450, 450, 450)
volume_creator_manager.set_element_sizes(0.5, 0.5, 0.5, "mm")
volume_creator_manager.set_output('data/volume.mhd')
volume_creator_manager.set_range_output('data/range_volume.txt')
volume_creator_manager.set_material('Air')
volume_creator_manager.set_data_type('MET_INT')
volume_creator_manager.initialize()

# ------------------------------------------------------------------------------
# STEP 4: Designing volume(s)
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
# STEP 5: Saving the final volume
volume_creator_manager.write()

# Printin RAM status
ram_manager.print_infos()

# Printing profiler summary
profiler_manager.print_summary_profile()

# ------------------------------------------------------------------------------
# STEP 6: Exit GGEMS safely
exit()
