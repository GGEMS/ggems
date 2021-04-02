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

parser.add_argument('-d', '--device', required=True, type=str, help="Set device type: all, cpu, gpu, gpu_nvidia, gpu_amd, gpu_intel", choices=['all', 'cpu', 'gpu', 'gpu_nvidia', 'gpu_amd', 'gpu_intel'])

args = parser.parse_args()

# Get arguments
device = args.device

# ------------------------------------------------------------------------------
# STEP 0: Level of verbosity during computation
GGEMSVerbosity(3)

# ------------------------------------------------------------------------------
# STEP 1: Selecting an OpenCL device
if device == "gpu_nvidia":
  opencl_manager.set_device_to_activate('gpu', 'nvidia');
elif device == "gpu_amd":
  opencl_manager.set_device_to_activate('gpu', 'amd');
elif device == "gpu_intel":
  opencl_manager.set_device_to_activate('gpu', 'intel');
else:
  opencl_manager.set_device_to_activate(device);

point_source = GGEMSXRaySource('point_source')
point_source.set_source_particle_type('gamma')
point_source.set_number_of_particles(10000000)
point_source.set_position(-595.0, 0.0, 0.0, 'mm')
point_source.set_rotation(0.0, 0.0, 0.0, 'deg')
point_source.set_beam_aperture(12.5, 'deg')
point_source.set_focal_spot_size(0.0, 0.0, 0.0, 'mm')
point_source.set_monoenergy(60.0, 'keV')

opencl_manager.print_infos()
#source_manager.initialize(123)
#ram_manager.print_infos()

# ------------------------------------------------------------------------------
# STEP 2: Exit safely
opencl_manager.clean()
exit()
