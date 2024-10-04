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
  prog='dosimetry_photon.py',
  description='-->> 4 - Dosimetry Example <<--',
  epilog='',
  formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('-d', '--device', required=False, type=str, default='all', help="OpenCL device (all, cpu, gpu, gpu_nvidia, gpu_intel, gpu_amd, X;Y;Z...)")
parser.add_argument('-b', '--balance', required=False, type=str, help="X;Y;Z... Balance computation for device if many devices are selected")
parser.add_argument('-n', '--nparticles', required=False, type=int, default=1000000, help="Number of particles")
parser.add_argument('-s', '--seed', required=False, type=int, default=777, help="Seed of pseudo generator number")
parser.add_argument('-v', '--verbose', required=False, type=int, default=0, help="Set level of verbosity")
parser.add_argument('-t', '--tle', required=False, action='store_true', help="Activating TLE method")

args = parser.parse_args()

# Get argument
device = args.device
verbosity_level = args.verbose
number_of_particles = args.nparticles
device_balancing = args.balance
seed = args.seed
is_tle = args.tle

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
# STEP 3: Setting GGEMS materials
materials_database_manager.set_materials('data/materials.txt')

# ------------------------------------------------------------------------------
# STEP 4: Phantoms

# Generating phantom
volume_creator_manager.set_dimensions(240, 240, 640)
volume_creator_manager.set_element_sizes(0.5, 0.5, 0.5, 'mm')
volume_creator_manager.set_output('data/phantom.mhd')
volume_creator_manager.set_range_output('data/range_phantom.txt')
volume_creator_manager.set_material('Air')
volume_creator_manager.set_data_type('MET_INT')
volume_creator_manager.initialize()

tube_phantom = GGEMSTube(50.0, 50.0, 300.0, 'mm')
tube_phantom.set_position(0.0, 0.0, 0.0, 'mm')
tube_phantom.set_label_value(1)
tube_phantom.set_material('Water')
tube_phantom.initialize()
tube_phantom.draw()
tube_phantom.delete()

volume_creator_manager.write()

# Loading phantom
phantom = GGEMSVoxelizedPhantom('phantom')
phantom.set_phantom('data/phantom.mhd', 'data/range_phantom.txt')
phantom.set_rotation(0.0, 0.0, 0.0, 'deg')
phantom.set_position(0.0, 0.0, 0.0, 'mm')

# ------------------------------------------------------------------------------
# STEP 5: Dosimetry
dosimetry = GGEMSDosimetryCalculator()
dosimetry.attach_to_navigator('phantom')
dosimetry.set_output_basename('data/dosimetry')
dosimetry.set_dosel_size(0.5, 0.5, 0.5, 'mm')
dosimetry.water_reference(False)
dosimetry.minimum_density(0.1, 'g/cm3')
dosimetry.set_tle(is_tle)

dosimetry.uncertainty(True)
dosimetry.photon_tracking(True)
dosimetry.edep(True)
dosimetry.hit(True)
dosimetry.edep_squared(True)

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
# STEP 6: Cuts, by default but are 1 um
range_cuts_manager.set_cut('gamma', 0.1, 'mm', 'all')

# ------------------------------------------------------------------------------
# STEP 7: Source
point_source = GGEMSXRaySource('point_source')
point_source.set_source_particle_type('gamma')
point_source.set_number_of_particles(number_of_particles)
point_source.set_position(-595.0, 0.0, 0.0, 'mm')
point_source.set_rotation(0.0, 0.0, 0.0, 'deg')
point_source.set_direction_type('isotropic')
point_source.set_beam_aperture(5.0, 'deg')
point_source.set_focal_spot_size(0.0, 0.0, 0.0, 'mm')
point_source.set_polyenergy('data/spectrum_120kVp_2mmAl.dat')

# ------------------------------------------------------------------------------
# STEP 8: GGEMS simulation
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
ggems.run()

# ------------------------------------------------------------------------------
# STEP 9: Exit safely
dosimetry.delete()
ggems.delete()
clean_safely()
exit()
