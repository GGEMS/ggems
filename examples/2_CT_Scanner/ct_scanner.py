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
GGEMSVerbosity(1)

# ------------------------------------------------------------------------------
# STEP 1: Choosing an OpenCL device
opencl_manager.set_device_index(device_id)

# ------------------------------------------------------------------------------
# STEP 2: Setting GGEMS materials
materials_database_manager.set_materials('../../data/materials.txt')

# ------------------------------------------------------------------------------
# STEP 3: Phantoms and systems

# Generating phantom
volume_creator_manager.set_dimensions(120, 120, 120)
volume_creator_manager.set_element_sizes(0.1, 0.1, 0.1, 'mm')
volume_creator_manager.set_output('data/phantom.mhd')
volume_creator_manager.set_range_output('data/range_phantom.txt')
volume_creator_manager.set_material('Air')
volume_creator_manager.set_data_type('MET_INT')
volume_creator_manager.initialize()

box_phantom = GGEMSBox(10.0, 10.0, 10.0, 'mm')
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
ct_detector.save('data/projection.mhd')

# ------------------------------------------------------------------------------
# STEP 4: Physics
processes_manager.add_process('Compton', 'gamma', 'all')
processes_manager.add_process('Photoelectric', 'gamma', 'all')
processes_manager.add_process('Rayleigh', 'gamma', 'all')

# Optional options, the following are by default
processes_manager.set_cross_section_table_number_of_bins(220)
processes_manager.set_cross_section_table_energy_min(1.0, 'keV')
processes_manager.set_cross_section_table_energy_max(1.0, 'MeV')

# ------------------------------------------------------------------------------
# STEP 5: Cuts, by default but are 1 um
range_cuts_manager.set_cut('gamma', 0.1, 'mm', 'all')

# ------------------------------------------------------------------------------
# STEP 6: Source
point_source = GGEMSXRaySource('point_source')
point_source.set_source_particle_type('gamma')
point_source.set_number_of_particles(10000000)
point_source.set_position(-595.0, 0.0, 0.0, 'mm')
point_source.set_rotation(0.0, 0.0, 0.0, 'deg')
point_source.set_beam_aperture(12.5, 'deg')
point_source.set_focal_spot_size(0.0, 0.0, 0.0, 'mm')
point_source.set_polyenergy('data/spectrum_120kVp_2mmAl.dat')

# ------------------------------------------------------------------------------
# STEP 7: GGEMS simulation
ggems_manager.opencl_verbose(False)
ggems_manager.material_database_verbose(False)
ggems_manager.navigator_verbose(True)
ggems_manager.source_verbose(True)
ggems_manager.memory_verbose(True)
ggems_manager.process_verbose(True)
ggems_manager.range_cuts_verbose(True)
ggems_manager.random_verbose(True)
ggems_manager.profiling_verbose(True)
ggems_manager.tracking_verbose(False, 0)

# Initializing the GGEMS simulation
ggems_manager.initialize()

# Start GGEMS simulation
ggems_manager.run()

# ------------------------------------------------------------------------------
# STEP 8: Exit safely
opencl_manager.clean()
exit()
