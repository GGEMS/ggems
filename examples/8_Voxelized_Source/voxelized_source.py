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
  prog='voxelized_source.py',
  description='-->> 8 - Voxelized Source for SPECT example <<--',
  epilog='',
  formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('-d', '--device', required=False, type=str, default='0', help="OpenCL device (all, cpu, gpu, gpu_nvidia, gpu_intel, gpu_amd, X;Y;Z...)")
parser.add_argument('-b', '--balance', required=False, type=str, help="X;Y;Z... Balance computation for device if many devices are selected. -b \"0.5;0.5\" means 50 %% of computation on device 0, and 50 %% of computation on device 1")
parser.add_argument('-s', '--seed', required=False, type=int, default=777, help="Seed of pseudo generator number")
parser.add_argument('-v', '--verbose', required=False, type=int, default=0, help="Set level of verbosity")
parser.add_argument('-e', '--nogl', required=False, action='store_false', help='Disable OpenGL')
parser.add_argument('-p', '--nparticlesgl', required=False, type=int, default=256, help='Number of displayed primary particles on OpenGL window (max: 65536)')
parser.add_argument('-c', '--drawgeom', required=False, action='store_true', help='Draw geometry only on OpenGL window')
parser.add_argument('-n', '--nparticles', required=False, type=int, default=1000000, help="Number of particles")

args = parser.parse_args()

# Get argument
device = args.device
verbosity_level = args.verbose
device_balancing = args.balance
seed = args.seed
number_of_displayed_particles = args.nparticlesgl
is_draw_geom = args.drawgeom
is_gl = args.nogl
number_of_particles = args.nparticles

# ------------------------------------------------------------------------------
# STEP 0: Level of verbosity during computation
GGEMSVerbosity(verbosity_level)

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
# STEP 2: Choosing an OpenCL device
if device == 'gpu_nvidia':
  opencl_manager.set_device_to_activate('gpu', 'nvidia')
elif device == 'gpu_amd':
  opencl_manager.set_device_to_activate('gpu', 'amd')
elif device == 'gpu_intel':
  opencl_manager.set_device_to_activate('gpu', 'intel')
else:
  opencl_manager.set_device_to_activate(device)

if (device_balancing):
  opencl_manager.set_device_balancing(device_balancing)

# ------------------------------------------------------------------------------
# STEP 3: Params for visualization
if is_gl:
  opengl_manager = GGEMSOpenGLManager()
  opengl_manager.set_window_dimensions(800, 800)
  opengl_manager.set_msaa(8)
  opengl_manager.set_background_color('black')
  opengl_manager.set_draw_axis(True)
  opengl_manager.set_world_size(3.0, 3.0, 3.0, 'm')
  opengl_manager.set_image_output('data/axis')
  opengl_manager.set_displayed_particles(number_of_displayed_particles)
  opengl_manager.initialize()

# ------------------------------------------------------------------------------
# STEP 4: Setting GGEMS materials
materials_database_manager.set_materials('data/materials.txt')

# ------------------------------------------------------------------------------
# STEP 5: Phantoms and systems

# Generating ATTENUATION phantom
volume_creator_manager.set_dimensions(200, 200, 150)
volume_creator_manager.set_element_sizes(1.0, 1.0, 1.0, 'mm')
volume_creator_manager.set_output('data/phantom_atn.mhd')
volume_creator_manager.set_range_output('data/range_phantom.txt')
volume_creator_manager.set_material('Air')
volume_creator_manager.set_data_type('MET_INT')
volume_creator_manager.initialize()

# Block (Water)
block_phantom = GGEMSTube(80.0, 80.0, 160.0, 'mm')
block_phantom.set_position(0.0, 0.0, 0.0, 'mm')
block_phantom.set_label_value(1)
block_phantom.set_material('Water')
block_phantom.initialize()
block_phantom.draw()
block_phantom.delete()

# Insert 1 (RibBone)
insert1_phantom = GGEMSTube(20.0, 20.0, 160.0, 'mm')
insert1_phantom.set_position(-40.0, 0.0, 0.0, 'mm')
insert1_phantom.set_label_value(2)
insert1_phantom.set_material('RibBone')
insert1_phantom.initialize()
insert1_phantom.draw()
insert1_phantom.delete()

# Insert 2 (Lung)
insert2_phantom = GGEMSTube(20.0, 20.0, 160.0, 'mm')
insert2_phantom.set_position(40.0, 0.0, 0.0, 'mm')
insert2_phantom.set_label_value(3)
insert2_phantom.set_material('Lung')
insert2_phantom.initialize()
insert2_phantom.draw()
insert2_phantom.delete()

volume_creator_manager.write()
volume_creator_manager.clean()

# Generating SOURCE phantom
volume_creator_manager.set_dimensions(200, 200, 150)
volume_creator_manager.set_element_sizes(1.0, 1.0, 1.0, 'mm')
volume_creator_manager.set_output('data/phantom_src.mhd')
volume_creator_manager.set_data_type('MET_FLOAT')
volume_creator_manager.initialize()

# Block (Water)
block_src_phantom = GGEMSTube(80.0, 80.0, 160.0, 'mm')
block_src_phantom.set_position(0.0, 0.0, 0.0, 'mm')
block_src_phantom.set_label_value(8.25) # Bq
block_src_phantom.initialize()
block_src_phantom.draw()
block_src_phantom.delete()

# Insert 1 (RibBone)
insert1_src_phantom = GGEMSTube(20.0, 20.0, 160.0, 'mm')
insert1_src_phantom.set_position(-40.0, 0.0, 0.0, 'mm')
insert1_src_phantom.set_label_value(20.36) # Bq
insert1_src_phantom.initialize()
insert1_src_phantom.draw()
insert1_src_phantom.delete()

# Insert 2 (Lung)
insert2_src_phantom = GGEMSTube(20.0, 20.0, 160.0, 'mm')
insert2_src_phantom.set_position(40.0, 0.0, 0.0, 'mm')
insert2_src_phantom.set_label_value(1.215) # Bq
insert2_src_phantom.initialize()
insert2_src_phantom.draw()
insert2_src_phantom.delete()

volume_creator_manager.write()
volume_creator_manager.clean()

# Loading phantom in GGEMS
phantom = GGEMSVoxelizedPhantom('phantom')
phantom.set_phantom('data/phantom_atn.mhd', 'data/range_phantom.txt')
phantom.set_rotation(0.0, 0.0, 0.0, 'deg')
phantom.set_position(0.0, 0.0, 0.0, 'mm')
phantom.set_visible(True)

# Loading collimator phantom
mesh_collimator = GGEMSMeshedPhantom('colli_mesh')
mesh_collimator.set_phantom('data/ColimatorMEGP-GENM670.stl')
mesh_collimator.set_rotation(0.0, 90.0, 0.0, 'deg')
mesh_collimator.set_position(870.0, 0.0, 0.0, 'mm')
mesh_collimator.set_mesh_octree_depth(4)
mesh_collimator.set_visible(True)
mesh_collimator.set_material('Lead')
mesh_collimator.set_material_color('Lead', color_name='yellow')

# Creating a planar detector
planar_detector = GGEMSCTSystem('custom')
planar_detector.set_ct_type('flat')
planar_detector.set_number_of_modules(1, 1)
planar_detector.set_number_of_detection_elements(100, 100, 1)
planar_detector.set_size_of_detection_elements(4.0, 4.0, 10.0, 'mm')
planar_detector.set_material('GOS')
planar_detector.set_source_detector_distance(905.0, 'mm')
planar_detector.set_source_isocenter_distance(0.0, 'mm')
planar_detector.set_rotation(0.0, 0.0, 0.0, 'deg')
planar_detector.set_global_system_position(0.0, 0.0, 0.0, 'mm')
planar_detector.set_threshold(100.0, 'keV')
planar_detector.save('data/projection')
planar_detector.store_scatter(True)
planar_detector.set_visible(True)
planar_detector.set_material_color('GOS', color_name='red')

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
vox_source = GGEMSVoxelizedSource('vox_source')
vox_source.set_phantom_source('data/phantom_src.mhd')
vox_source.set_number_of_particles(number_of_particles)
vox_source.set_source_particle_type('gamma')
vox_source.set_position(0.0, 0.0, 0.0, 'mm')
#vox_source.set_monoenergy(140.51, 'keV')
vox_source.set_energy_peak(321.3, 'keV', 0.0021)
vox_source.set_energy_peak(249.7, 'keV', 0.0020)
vox_source.set_energy_peak(112.9, 'keV', 0.0617)
vox_source.set_energy_peak( 71.6, 'keV', 0.0017)
vox_source.set_energy_peak(208.4, 'keV', 0.1036)
vox_source.set_energy_peak(136.7, 'keV', 0.0005)

# ------------------------------------------------------------------------------
# STEP 9: GGEMS simulation
ggems = GGEMS()
ggems.opencl_verbose(True)
ggems.material_database_verbose(False)
ggems.navigator_verbose(True)
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
exit()
