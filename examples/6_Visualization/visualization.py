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

parser.add_argument('-v', '--verbose', required=False, type=int, default=0, help='Set level of verbosity')
parser.add_argument('-e', '--nogl', required=False, action='store_false', help='Disable OpenGL')
parser.add_argument('-w', '--wdims', required=False, default=[800, 800], type=int, nargs=2, help='OpenGL window dimensions')
parser.add_argument('-m', '--msaa', required=False, type=int, default=8, help='MSAA factor (1x, 2x, 4x or 8x)')
parser.add_argument('-a', '--axis', required=False, action='store_true', help='Drawing axis in OpenGL window')
parser.add_argument('-p', '--nparticlesgl', required=False, type=int, default=256, help='Number of displayed primary particles on OpenGL window (max: 65536)')
parser.add_argument('-b', '--drawgeom', required=False, action='store_true', help='Draw geometry only on OpenGL window')
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
is_draw_geom = args.drawgeom
is_gl = args.nogl

# ------------------------------------------------------------------------------
# STEP 0: Level of verbosity during computation
GGEMSVerbosity(verbosity_level)

# ------------------------------------------------------------------------------
# STEP 1: Calling C++ singleton
opencl_manager = GGEMSOpenCLManager()
materials_database_manager = GGEMSMaterialsDatabaseManager()
processes_manager = GGEMSProcessesManager()
range_cuts_manager = GGEMSRangeCutsManager()
volume_creator_manager = GGEMSVolumeCreatorManager()

# ------------------------------------------------------------------------------
# STEP 2: Params for visualization
if is_gl:
  opengl_manager = GGEMSOpenGLManager()
  opengl_manager.set_window_dimensions(window_dims[0], window_dims[1])
  opengl_manager.set_msaa(msaa)
  opengl_manager.set_background_color(window_color)
  opengl_manager.set_draw_axis(is_axis)
  opengl_manager.set_world_size(3.0, 3.0, 3.0, 'm')
  opengl_manager.set_image_output('data/axis')
  opengl_manager.set_displayed_particles(number_of_displayed_particles)
  opengl_manager.set_particle_color('gamma', 152, 251, 152)
  # opengl_manager.set_particle_color('gamma', color_name='red') # Using registered color
  opengl_manager.initialize()

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
phantom.set_material_visible('Air', True)
phantom.set_material_color('Water', color_name='blue') # Uncomment for automatic color

cbct_detector = GGEMSCTSystem('custom')
cbct_detector.set_ct_type('flat')
cbct_detector.set_number_of_modules(1, 1)
cbct_detector.set_number_of_detection_elements(400, 400, 1)
cbct_detector.set_size_of_detection_elements(1.0, 1.0, 10.0, 'mm')
cbct_detector.set_material('GOS')
cbct_detector.set_source_detector_distance(1500.5, 'mm') # Center of inside detector, adding half of detector (= SDD surface + 10.0/2 mm half of depth)
cbct_detector.set_source_isocenter_distance(900.0, 'mm')
cbct_detector.set_rotation(0.0, 0.0, 0.0, 'deg')
cbct_detector.set_global_system_position(0.0, 0.0, 0.0, 'mm');
cbct_detector.set_threshold(10.0, 'keV')
cbct_detector.save('data/projection')
cbct_detector.store_scatter(True)
cbct_detector.set_visible(True)
cbct_detector.set_material_color('GOS', 255, 0, 0) # Custom color using RGB
#cbct_detector.set_material_color('GOS', color_name='red') # Using registered color

cbct_detector2 = GGEMSCTSystem('custom2')
cbct_detector2.set_ct_type('flat')
cbct_detector2.set_number_of_modules(1, 1)
cbct_detector2.set_number_of_detection_elements(400, 400, 1)
cbct_detector2.set_size_of_detection_elements(1.0, 1.0, 10.0, 'mm')
cbct_detector2.set_material('Silicon')
cbct_detector2.set_source_detector_distance(1605.0, 'mm') # Center of inside detector, adding half of detector (= SDD surface + 10.0/2 mm half of depth)
cbct_detector2.set_source_isocenter_distance(1200.0, 'mm')
cbct_detector2.set_rotation(0.0, 0.0, 90.0, 'deg')
cbct_detector2.set_global_system_position(0.0, 0.0, 0.0, 'mm');
cbct_detector2.set_threshold(10.0, 'keV')
cbct_detector2.save('data/projection2')
cbct_detector2.store_scatter(True)
cbct_detector2.set_visible(True)

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
point_source.set_position(-900.0, 0.0, 0.0, 'mm')
point_source.set_rotation(0.0, 0.0, 0.0, 'deg')
point_source.set_beam_aperture(12.5, 'deg')
point_source.set_focal_spot_size(0.2, 0.6, 0.0, 'mm')
point_source.set_polyenergy('data/spectrum_120kVp_2mmAl.dat')

point_source2 = GGEMSXRaySource('point_source2')
point_source2.set_source_particle_type('gamma')
point_source2.set_number_of_particles(number_of_particles)
point_source2.set_position(-1200.0, 0.0, 0.0, 'mm')
point_source2.set_rotation(0.0, 0.0, 90.0, 'deg')
point_source2.set_beam_aperture(8.5, 'deg')
point_source2.set_focal_spot_size(0.2, 0.6, 0.0, 'mm')
point_source2.set_polyenergy('data/spectrum_120kVp_2mmAl.dat')

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

if is_draw_geom and is_gl: # Draw only geometry and do not run GGEMS
  opengl_manager.display()
else: # Running GGEMS and draw particles
  ggems.run()

# ------------------------------------------------------------------------------
# STEP 10: Exit safely
clean_safely()
exit()
