from ggems_lib import *


class GGEMSSourceManager(object):
    """Class managing source in GGEMS
    """
    def __init__(self):
        ggems_lib.get_instance_ggems_source_manager.restype = ctypes.c_void_p

        ggems_lib.print_infos_ggems_source_manager.argtypes = [ctypes.c_void_p]
        ggems_lib.print_infos_ggems_source_manager.restype = ctypes.c_void_p

        self.obj = ggems_lib.get_instance_ggems_source_manager()

    def print_infos(self):
        ggems_lib.print_infos_ggems_source_manager(self.obj)


class GGEMSXRaySource(object):
    """GGEMS XRay source class managing source for CT/CBCT simulation
    """
    def __init__(self):
        ggems_lib.create_ggems_xray_source.restype = ctypes.c_void_p

        ggems_lib.initialize_ggems_xray_source.argtypes = [ctypes.c_void_p]
        ggems_lib.initialize_ggems_xray_source.restype = ctypes.c_void_p

        ggems_lib.set_source_name_ggems_xray_source.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_source_name_ggems_xray_source.restype = ctypes.c_void_p
    
        ggems_lib.set_position_ggems_xray_source.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_position_ggems_xray_source.restype = ctypes.c_void_p

        ggems_lib.set_number_of_particles_xray_source.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong]
        ggems_lib.set_number_of_particles_xray_source.restype = ctypes.c_void_p

        ggems_lib.set_source_particle_type_ggems_xray_source.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_source_particle_type_ggems_xray_source.restype = ctypes.c_void_p

        ggems_lib.set_beam_aperture_ggems_xray_source.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_beam_aperture_ggems_xray_source.restype = ctypes.c_void_p

        ggems_lib.set_focal_spot_size_ggems_xray_source.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_focal_spot_size_ggems_xray_source.restype = ctypes.c_void_p

        ggems_lib.set_local_axis_ggems_xray_source.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
        ggems_lib.set_local_axis_ggems_xray_source.restype = ctypes.c_void_p

        ggems_lib.set_rotation_ggems_xray_source.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_rotation_ggems_xray_source.restype = ctypes.c_void_p

        ggems_lib.set_monoenergy_ggems_xray_source.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_char_p]
        ggems_lib.set_monoenergy_ggems_xray_source.restype = ctypes.c_void_p

        ggems_lib.set_polyenergy_ggems_xray_source.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_polyenergy_ggems_xray_source.restype = ctypes.c_void_p

        self.obj = ggems_lib.create_ggems_xray_source()

    def initialize(self):
        ggems_lib.initialize_ggems_xray_source(self.obj)

    def set_source_name(self, name):
        ggems_lib.set_source_name_ggems_xray_source(self.obj, name.encode('ASCII'))

    def set_position(self, x, y, z, unit):
        ggems_lib.set_position_ggems_xray_source(self.obj, x, y, z, unit.encode('ASCII'))

    def set_number_of_particles(self, number_of_particles):
        ggems_lib.set_number_of_particles_xray_source(self.obj, number_of_particles)

    def set_source_particle_type(self, particle_type):
        ggems_lib.set_source_particle_type_ggems_xray_source(self.obj, particle_type.encode('ASCII'))

    def set_beam_aperture(self, beam_aperture, unit):
        ggems_lib.set_beam_aperture_ggems_xray_source(self.obj, beam_aperture, unit.encode('ASCII'))

    def set_focal_spot_size(self, width, height, depth, unit):
        ggems_lib.set_focal_spot_size_ggems_xray_source(self.obj, width, height, depth, unit.encode('ASCII'))

    def set_local_axis(self, m00, m01, m02, m10, m11, m12, m20, m21, m22):
        ggems_lib.set_local_axis_ggems_xray_source(self.obj, m00, m01, m02, m10, m11, m12, m20, m21, m22)

    def set_rotation(self, rx, ry, rz, unit):
        ggems_lib.set_rotation_ggems_xray_source(self.obj, rx, ry, rz, unit.encode('ASCII'))

    def set_monoenergy(self, e, unit):
        ggems_lib.set_monoenergy_ggems_xray_source(self.obj, e, unit.encode('ASCII'))

    def set_polyenergy(self, file):
        ggems_lib.set_polyenergy_ggems_xray_source(self.obj, file.encode('ASCII'))