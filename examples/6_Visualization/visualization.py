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
parser.add_argument('-v', '--verbose', required=False, type=int, default=0, help="Set level of verbosity")
args = parser.parse_args()

# Getting arguments
verbosity_level = args.verbose

# ------------------------------------------------------------------------------
# STEP 0: Level of verbosity during computation
GGEMSVerbosity(verbosity_level)

# ------------------------------------------------------------------------------
# STEP 1: Calling C++ singleton

opencl_manager = GGEMSOpenCLManager()

# ------------------------------------------------------------------------------
# STEP 2: Choosing an OpenCL device
opencl_manager.set_device_to_activate("all")
opencl_manager.print_infos()

# ------------------------------------------------------------------------------
# STEP X: Exit safely
clean_safely()
exit()