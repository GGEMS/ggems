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

        self.obj = ggems_lib.get_instance_ggems_opengl_manager()

    def set_window_dimensions(self, width, height):
        ggems_lib.set_window_dimensions_ggems_opengl_manager(self.obj, width, height)

    def set_background_color(self, color):
        ggems_lib.set_background_color_ggems_opengl_manager(self.obj, color.encode('ASCII'))

    def set_msaa(self, msaa):
        ggems_lib.set_msaa_ggems_opengl_manager(self.obj, msaa)
