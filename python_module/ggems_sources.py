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

from .ggems_lib import *

class GGEMSSourceManager(object):
    """Get the C++ singleton and initialize source
    """
    def __init__(self):
        ggems_lib.get_instance_ggems_source_manager.restype = ctypes.c_void_p

        ggems_lib.initialize_source_manager.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        ggems_lib.initialize_source_manager.restype = ctypes.c_void_p

        ggems_lib.print_infos_source_manager.argtypes = [ctypes.c_void_p]
        ggems_lib.print_infos_source_manager.restype = ctypes.c_void_p

        ggems_lib.clean_source_manager.argtypes = [ctypes.c_void_p]
        ggems_lib.clean_source_manager.restype = ctypes.c_void_p

        self.obj = ggems_lib.get_instance_ggems_source_manager()

    def initialize(self, seed):
        ggems_lib.initialize_source_manager(self.obj, seed)

    def print_infos(self):
        ggems_lib.print_infos_source_manager(self.obj)

    def clean(self):
        ggems_lib.clean_source_manager(self.obj)


class GGEMSXRaySource(object):
  """GGEMS XRay source class managing source for CT/CBCT simulation
  """
  def __init__(self, source_name):
      ggems_lib.create_ggems_xray_source.argtypes = [ctypes.c_char_p]
      ggems_lib.create_ggems_xray_source.restype = ctypes.c_void_p
    
      ggems_lib.set_position_ggems_xray_source.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_char_p]
      ggems_lib.set_position_ggems_xray_source.restype = ctypes.c_void_p

      ggems_lib.set_number_of_particles_xray_source.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
      ggems_lib.set_number_of_particles_xray_source.restype = ctypes.c_void_p

      ggems_lib.set_source_particle_type_ggems_xray_source.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
      ggems_lib.set_source_particle_type_ggems_xray_source.restype = ctypes.c_void_p

      ggems_lib.set_source_direction_type_ggems_xray_source.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
      ggems_lib.set_source_direction_type_ggems_xray_source.restype = ctypes.c_void_p

      ggems_lib.set_beam_aperture_ggems_xray_source.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_char_p]
      ggems_lib.set_beam_aperture_ggems_xray_source.restype = ctypes.c_void_p

      ggems_lib.set_focal_spot_size_ggems_xray_source.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_char_p]
      ggems_lib.set_focal_spot_size_ggems_xray_source.restype = ctypes.c_void_p

      ggems_lib.set_rotation_ggems_xray_source.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_char_p]
      ggems_lib.set_rotation_ggems_xray_source.restype = ctypes.c_void_p

      ggems_lib.set_monoenergy_ggems_xray_source.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_char_p]
      ggems_lib.set_monoenergy_ggems_xray_source.restype = ctypes.c_void_p

      ggems_lib.set_polyenergy_ggems_xray_source.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
      ggems_lib.set_polyenergy_ggems_xray_source.restype = ctypes.c_void_p

      ggems_lib.set_theta_histogram_ggems_xray_source.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_char_p]
      ggems_lib.set_theta_histogram_ggems_xray_source.restype = ctypes.c_void_p

      ggems_lib.set_phi_histogram_ggems_xray_source.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_char_p]
      ggems_lib.set_phi_histogram_ggems_xray_source.restype = ctypes.c_void_p

      self.obj = ggems_lib.create_ggems_xray_source(source_name.encode('ASCII'))

  def set_position(self, x, y, z, unit):
      ggems_lib.set_position_ggems_xray_source(self.obj, x, y, z, unit.encode('ASCII'))

  def set_number_of_particles(self, number_of_particles):
      ggems_lib.set_number_of_particles_xray_source(self.obj, number_of_particles)

  def set_source_particle_type(self, particle_type):
      ggems_lib.set_source_particle_type_ggems_xray_source(self.obj, particle_type.encode('ASCII'))

  def set_direction_type(self, direction_type):
      ggems_lib.set_source_direction_type_ggems_xray_source(self.obj, direction_type.encode('ASCII'))

  def set_beam_aperture(self, beam_aperture, unit):
      ggems_lib.set_beam_aperture_ggems_xray_source(self.obj, beam_aperture, unit.encode('ASCII'))

  def set_focal_spot_size(self, width, height, depth, unit):
      ggems_lib.set_focal_spot_size_ggems_xray_source(self.obj, width, height, depth, unit.encode('ASCII'))

  def set_rotation(self, rx, ry, rz, unit):
      ggems_lib.set_rotation_ggems_xray_source(self.obj, rx, ry, rz, unit.encode('ASCII'))

  def set_monoenergy(self, e, unit):
      ggems_lib.set_monoenergy_ggems_xray_source(self.obj, e, unit.encode('ASCII'))

  def set_polyenergy(self, file):
      ggems_lib.set_polyenergy_ggems_xray_source(self.obj, file.encode('ASCII'))

  def set_theta_histogram(self, theta_weights, theta_angles, unit):
      ggems_lib.set_theta_histogram_ggems_xray_source(self.obj, (ctypes.c_float * len(theta_weights))(*theta_weights), (ctypes.c_float * len(theta_angles))(*theta_angles), len(theta_weights),unit.encode('ASCII'))

  def set_phi_histogram(self, phi_weights, phi_angles, unit):
      ggems_lib.set_phi_histogram_ggems_xray_source(self.obj, (ctypes.c_float * len(phi_weights))(*phi_weights), (ctypes.c_float * len(phi_angles))(*phi_angles), len(phi_weights),unit.encode('ASCII'))