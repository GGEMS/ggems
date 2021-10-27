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
parser.add_argument('-s', '--seed', required=False, type=int, default=777, help="Seed of pseudo generator number")
parser.add_argument('-o', '--ogl', required=False, action='store_true', help="Activating OpenGL visu")
parser.add_argument('-d', '--device', required=False, type=str, default='0', help="OpenCL device running visualization")
args = parser.parse_args()

# Getting arguments
verbosity_level = args.verbose
seed = args.seed
is_ogl = args.ogl
device = args.device

# ------------------------------------------------------------------------------
# STEP 0: Level of verbosity during computation
GGEMSVerbosity(verbosity_level)

# ------------------------------------------------------------------------------
# STEP 1: Calling C++ singleton
opengl_manager = GGEMSOpenGLManager()
opencl_manager = GGEMSOpenCLManager()
materials_database_manager = GGEMSMaterialsDatabaseManager()

# ------------------------------------------------------------------------------
# STEP 2: Params for visualization
opengl_manager.set_window_dimensions(1200, 800)
opengl_manager.set_msaa(8)
opengl_manager.set_background_color('black')
opengl_manager.set_draw_axis(True)
#opengl_manager.set_particle_color('gamma','black')
#opengl_manager.set_mode_view('perspective') #ortho
#opengl_manager.set_max_particle(1000)
#opengl_manager.save('') #image or movie
# Param de l'angle de vue...

# ------------------------------------------------------------------------------
# STEP 3: Choosing an OpenCL device
opencl_manager.set_device_to_activate(device)

# ------------------------------------------------------------------------------
# STEP 4: Setting GGEMS materials
materials_database_manager.set_materials('data/materials.txt')

# ------------------------------------------------------------------------------
# STEP 5: Phantoms and systems
# Creating a CBCT detector
cbct_detector = GGEMSCTSystem('custom')
cbct_detector.set_ct_type('flat')
cbct_detector.set_number_of_modules(1, 1)
cbct_detector.set_number_of_detection_elements(400, 400, 1)
cbct_detector.set_size_of_detection_elements(1.0, 1.0, 10.0, 'mm')
cbct_detector.set_material('Silicon')
cbct_detector.set_source_detector_distance(1500.0, 'mm')
cbct_detector.set_source_isocenter_distance(900.0, 'mm')
cbct_detector.set_rotation(0.0, 0.0, 0.0, 'deg')
cbct_detector.set_threshold(10.0, 'keV')
cbct_detector.save('data/projection.mhd')
#cbct_detector.set_color('black')
#cbct_detector.set (plein or contour(line))
#cbct_detector.set_visible(false/true)

# ------------------------------------------------------------------------------
# STEP 6: GGEMS simulation
ggems = GGEMS(is_ogl)
ggems.opencl_verbose(False)
ggems.material_database_verbose(False)
ggems.navigator_verbose(False)
ggems.source_verbose(False)
ggems.memory_verbose(False)
ggems.process_verbose(False)
ggems.range_cuts_verbose(False)
ggems.random_verbose(False)
ggems.profiling_verbose(False)
ggems.tracking_verbose(False, 0)

# Initializing the GGEMS simulation
ggems.initialize(seed)

# Start GGEMS simulation
ggems.run()

# ------------------------------------------------------------------------------
# STEP 7: Exit safely
clean_safely()
exit()
