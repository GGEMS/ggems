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

from ggems_lib import *

class GGEMSOpenGLManager(object):
    """Get the OpenGL C++ singleton and print infos or managing it
    """
    def __init__(self):
        ggems_lib.get_instance_ggems_opengl_manager.restype = ctypes.c_void_p

        ggems_lib.set_msaa_ggems_opengl_manager.argtypes = [ctypes.c_void_p, ctypes.c_int]
        ggems_lib.set_msaa_ggems_opengl_manager.restype = ctypes.c_void_p

        ggems_lib.set_background_color_ggems_opengl_manager.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_background_color_ggems_opengl_manager.restype = ctypes.c_void_p

        ggems_lib.set_window_dimensions_ggems_opengl_manager.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        ggems_lib.set_window_dimensions_ggems_opengl_manager.restype = ctypes.c_void_p

        ggems_lib.set_world_size_ggems_opengl_manager.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_world_size_ggems_opengl_manager.restype = ctypes.c_void_p

        ggems_lib.initialize_ggems_opengl_manager.argtypes = [ctypes.c_void_p]
        ggems_lib.initialize_ggems_opengl_manager.restype = ctypes.c_void_p

        ggems_lib.display_ggems_opengl_manager.argtypes = [ctypes.c_void_p]
        ggems_lib.display_ggems_opengl_manager.restype = ctypes.c_void_p

        ggems_lib.set_displayed_particles_ggems_opengl_manager.argtypes = [ctypes.c_void_p, ctypes.c_int]
        ggems_lib.set_displayed_particles_ggems_opengl_manager.restype = ctypes.c_void_p

        ggems_lib.set_image_output_ggems_opengl_manager.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_image_output_ggems_opengl_manager.restype = ctypes.c_void_p

        ggems_lib.set_particle_color_ggems_opengl_manager.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_ubyte]
        ggems_lib.set_particle_color_ggems_opengl_manager.restype = ctypes.c_void_p

        ggems_lib.set_particle_color_name_ggems_opengl_manager.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        ggems_lib.set_particle_color_name_ggems_opengl_manager.restype = ctypes.c_void_p

        ggems_lib.set_draw_axis_ggems_opengl_manager.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        ggems_lib.set_draw_axis_ggems_opengl_manager.restype = ctypes.c_void_p

        self.obj = ggems_lib.get_instance_ggems_opengl_manager()

    def set_window_dimensions(self, width, height):
        ggems_lib.set_window_dimensions_ggems_opengl_manager(self.obj, width, height)

    def set_background_color(self, color):
        ggems_lib.set_background_color_ggems_opengl_manager(self.obj, color.encode('ASCII'))

    def set_draw_axis(self, color):
        ggems_lib.set_draw_axis_ggems_opengl_manager(self.obj, color)

    def set_world_size(self, width, height, depth, unit):
        ggems_lib.set_world_size_ggems_opengl_manager(self.obj, width, height, depth, unit.encode('ASCII'))

    def set_image_output(self, path):
        ggems_lib.set_image_output_ggems_opengl_manager(self.obj, path.encode('ASCII'))

    def initialize(self):
        ggems_lib.initialize_ggems_opengl_manager(self.obj)

    def display(self):
        ggems_lib.display_ggems_opengl_manager(self.obj)

    def set_displayed_particles(self, particles):
        ggems_lib.set_displayed_particles_ggems_opengl_manager(self.obj, particles)

    def set_particle_color(self, particle_type, red=0, green=0, blue=0, color_name=''):
        if color_name:
            ggems_lib.set_particle_color_name_ggems_opengl_manager(self.obj, particle_type.encode('ASCII'), color_name.encode('ASCII'))
        else:
            ggems_lib.set_particle_color_ggems_opengl_manager(self.obj, particle_type.encode('ASCII'), red, green, blue)

    def set_msaa(self, msaa):
        ggems_lib.set_msaa_ggems_opengl_manager(self.obj, msaa)
