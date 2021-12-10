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
  prog='visualization.py',
  description='''-->> 6 - OpenGL Visualization Example <<--''',
  epilog='''''',
  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-v', '--verbose', required=False, type=int, default=0, help="Set level of verbosity")
parser.add_argument('-w', '--wdims', required=False, default=[800, 800], type=int, nargs=2, help='OpenGL window dimensions')
parser.add_argument('-m', '--msaa', required=False, type=int, default=8, help="MSAA factor (1x, 2x, 4x or 8x)")
parser.add_argument('-a', '--axis', required=False, action='store_true', help="Drawing axis in OpenGL window")
parser.add_argument('-p', '--nparticlesgl', required=False, type=int, default=1024, help="Number of displayed primary particles on OpenGL window")
parser.add_argument('-c', '--wcolor', required=False, type=str, default='black', help='Background color of OpenGL window')
parser.add_argument('-d', '--device', required=False, type=str, default='0', help="OpenCL device running visualization")
parser.add_argument('-n', '--nparticles', required=False, type=int, default=1000000, help="Number of particles")
parser.add_argument('-s', '--seed', required=False, type=int, default=777, help="Seed of pseudo generator number")

args = parser.parse_args()

# Getting arguments
verbosity_level = args.verbose
seed = args.seed
device = args.device
number_of_particles = args.nparticles
number_of_displayed_particles = args.nparticlesgl
is_axis = args.axis
msaa = args.msaa
window_color = args.wcolor
window_dims = args.wdims

# ------------------------------------------------------------------------------
# STEP 0: Level of verbosity during computation
GGEMSVerbosity(verbosity_level)

# ------------------------------------------------------------------------------
# STEP 1: Calling C++ singleton
opengl_manager = GGEMSOpenGLManager()
opencl_manager = GGEMSOpenCLManager()
materials_database_manager = GGEMSMaterialsDatabaseManager()
processes_manager = GGEMSProcessesManager()
range_cuts_manager = GGEMSRangeCutsManager()
volume_creator_manager = GGEMSVolumeCreatorManager()

# ------------------------------------------------------------------------------
# STEP 2: Params for visualization
opengl_manager.set_window_dimensions(window_dims[0], window_dims[1])
opengl_manager.set_msaa(msaa)
opengl_manager.set_background_color(window_color)
opengl_manager.set_draw_axis(is_axis)
opengl_manager.set_world_size(2.0, 2.0, 2.0, 'm')
opengl_manager.set_image_output('data/axis')
opengl_manager.set_displayed_particles(number_of_displayed_particles) # Not exceed 4096!!!
opengl_manager.initialize();

# ------------------------------------------------------------------------------
# STEP 3: Choosing an OpenCL device
opencl_manager.set_device_to_activate(device)

# ------------------------------------------------------------------------------
# STEP 4: Setting GGEMS materials
materials_database_manager.set_materials('data/materials.txt')

# ------------------------------------------------------------------------------
# STEP 5: Phantoms and systems

# Generating phantom
volume_creator_manager.set_dimensions(120, 120, 120)
volume_creator_manager.set_element_sizes(1.0, 1.0, 1.0, 'mm')
volume_creator_manager.set_output('data/phantom.mhd')
volume_creator_manager.set_range_output('data/range_phantom.txt')
volume_creator_manager.set_material('Air')
volume_creator_manager.set_data_type('MET_INT')
volume_creator_manager.initialize()

box_phantom = GGEMSBox(80.0, 80.0, 80.0, 'mm')
box_phantom.set_position(0.0, 0.0, 0.0, 'mm')
box_phantom.set_label_value(1)
box_phantom.set_material('Water')
box_phantom.initialize()
box_phantom.draw()
box_phantom.delete()

volume_creator_manager.write()

# Loading phantom in GGEMS
phantom = GGEMSVoxelizedPhantom('phantom')
phantom.set_phantom('data/phantom.mhd', 'data/range_phantom.txt')
phantom.set_rotation(0.0, 0.0, 0.0, 'deg')
phantom.set_position(0.0, 0.0, 0.0, 'mm')
phantom.set_visible(True)
phantom.set_material_visible('Air', False)
phantom.set_material_color('Water', color_name='blue') # Uncomment for automatic color

ct_detector = GGEMSCTSystem('Stellar')
ct_detector.set_ct_type('curved')
ct_detector.set_number_of_modules(1, 46)
ct_detector.set_number_of_detection_elements(64, 16, 1)
ct_detector.set_size_of_detection_elements(0.6, 0.6, 0.6, 'mm')
ct_detector.set_material('GOS')
ct_detector.set_source_detector_distance(1085.6, 'mm') 
ct_detector.set_source_isocenter_distance(595.0, 'mm')
ct_detector.set_rotation(0.0, 0.0, 0.0, 'deg')
ct_detector.set_threshold(10.0, 'keV')
ct_detector.save('data/projection')
ct_detector.store_scatter(True)
ct_detector.set_visible(True)
ct_detector.set_material_color("GOS", 255, 0, 0) # Custom color using RGB
# ct_detector.set_material_color("GOS", 'red') # Using registered color

# ------------------------------------------------------------------------------
# STEP 6: Physics
processes_manager.add_process('Compton', 'gamma', 'all')
processes_manager.add_process('Photoelectric', 'gamma', 'all')
processes_manager.add_process('Rayleigh', 'gamma', 'all')

# Optional options, the following are by default
processes_manager.set_cross_section_table_number_of_bins(220)
processes_manager.set_cross_section_table_energy_min(1.0, 'keV')
processes_manager.set_cross_section_table_energy_max(1.0, 'MeV')

# ------------------------------------------------------------------------------
# STEP 7: Cuts, by default but are 1 um
range_cuts_manager.set_cut('gamma', 0.1, 'mm', 'all')

# ------------------------------------------------------------------------------
# STEP 8: Source
point_source = GGEMSXRaySource('point_source')
point_source.set_source_particle_type('gamma')
point_source.set_number_of_particles(number_of_particles)
point_source.set_position(-595.0, 0.0, 0.0, 'mm')
point_source.set_rotation(0.0, 0.0, 0.0, 'deg')
point_source.set_beam_aperture(12.5, 'deg')
point_source.set_focal_spot_size(0.0, 0.0, 0.0, 'mm')
point_source.set_polyenergy('data/spectrum_120kVp_2mmAl.dat')

# ------------------------------------------------------------------------------
# STEP 9: GGEMS simulation
ggems = GGEMS()
ggems.opencl_verbose(True)
ggems.material_database_verbose(False)
ggems.navigator_verbose(False)
ggems.source_verbose(True)
ggems.memory_verbose(True)
ggems.process_verbose(True)
ggems.range_cuts_verbose(True)
ggems.random_verbose(True)
ggems.profiling_verbose(True)
ggems.tracking_verbose(False, 0)

# Initializing the GGEMS simulation
ggems.initialize(seed)

# Start GGEMS simulation
# ggems.run()

opengl_manager.display()

# ------------------------------------------------------------------------------
# STEP 10: Exit safely
clean_safely()
exit()
