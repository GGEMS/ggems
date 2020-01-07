import ctypes
import sys

if sys.platform == "linux":
    ggems_lib = ctypes.cdll.LoadLibrary(
        "/home/dbenoit/data/Build/GGEMS_OpenCL/libggems.so")
elif sys.platform == "darwin":
    ggems_lib = ctypes.cdll.LoadLibrary(
        "/home/dbenoit/data/Build/GGEMS_OpenCL/libggems.dylib")
elif sys.platform == "win32":
    ggems_lib = ctypes.cdll.LoadLibrary(
        "C:\\Users\\dbenoit\\Workspace\\GGEMS_OpenCL_build\\libggems.dll")


class GGEMSOpenCLManager(object):
    """Get the OpenCL C++ singleton and print infos or managing it
    """
    def __init__(self):
        ggems_lib.get_instance_ggems_opencl_manager.restype = ctypes.c_void_p

        ggems_lib.print_platform_ggems_opencl_manager.argtypes = [
            ctypes.c_void_p]
        ggems_lib.print_platform_ggems_opencl_manager.restype = ctypes.c_void_p

        ggems_lib.print_device_ggems_opencl_manager.argtypes = [
            ctypes.c_void_p]
        ggems_lib.print_device_ggems_opencl_manager.restype = ctypes.c_void_p

        ggems_lib.print_build_options_ggems_opencl_manager.argtypes = [
            ctypes.c_void_p]
        ggems_lib.print_build_options_ggems_opencl_manager.restype =\
            ctypes.c_void_p

        ggems_lib.print_context_ggems_opencl_manager.argtypes = [
            ctypes.c_void_p]
        ggems_lib.print_context_ggems_opencl_manager.restype = ctypes.c_void_p

        ggems_lib.print_RAM_ggems_opencl_manager.argtypes = [ctypes.c_void_p]
        ggems_lib.print_RAM_ggems_opencl_manager.restype = ctypes.c_void_p

        ggems_lib.print_command_queue_ggems_opencl_manager.argtypes = [
            ctypes.c_void_p]
        ggems_lib.print_command_queue_ggems_opencl_manager.restype =\
            ctypes.c_void_p

        ggems_lib.set_context_index_ggems_opencl_manager.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32]
        ggems_lib.set_context_index_ggems_opencl_manager.restype =\
            ctypes.c_void_p

        ggems_lib.print_activated_context_ggems_opencl_manager.argtypes = [
            ctypes.c_void_p]
        ggems_lib.print_activated_context_ggems_opencl_manager.restype =\
            ctypes.c_void_p

        ggems_lib.clean_ggems_opencl_manager.argtypes = [ctypes.c_void_p]
        ggems_lib.clean_ggems_opencl_manager.restype = ctypes.c_void_p

        self.obj = ggems_lib.get_instance_ggems_opencl_manager()

    def print_infos(self):
        ggems_lib.print_platform_ggems_opencl_manager(self.obj)
        ggems_lib.print_device_ggems_opencl_manager(self.obj)
        ggems_lib.print_build_options_ggems_opencl_manager(self.obj)
        ggems_lib.print_context_ggems_opencl_manager(self.obj)
        ggems_lib.print_command_queue_ggems_opencl_manager(self.obj)
        ggems_lib.print_activated_context_ggems_opencl_manager(self.obj)

    def print_RAM(self):
        ggems_lib.print_RAM_ggems_opencl_manager(self.obj)

    def set_context_index(self, context_id):
        ggems_lib.set_context_index_ggems_opencl_manager(self.obj, context_id)

    def delete(self):
        ggems_lib.clean_ggems_opencl_manager(self.obj)


class GGEMSVerbosity(object):
    """Set the verbosity of infos in GGEMS
    """
    def __init__(self, val):
        ggems_lib.set_ggems_verbose.argtypes = [ctypes.c_int]
        ggems_lib.set_ggems_verbose.restype = ctypes.c_void_p

        ggems_lib.set_ggems_verbose(val)


class GGEMSXRaySource(object):
    """GGEMS XRay source class managing source for CT/CBCT simulation
    """
    def __init__(self):
        ggems_lib.create_ggems_xray_source.restype = ctypes.c_void_p

        ggems_lib.delete_ggems_xray_source.argtypes = [ctypes.c_void_p]
        ggems_lib.delete_ggems_xray_source.restype = ctypes.c_void_p

        ggems_lib.initialize_ggems_xray_source.argtypes = [ctypes.c_void_p]
        ggems_lib.initialize_ggems_xray_source.restype = ctypes.c_void_p

        ggems_lib.set_position_ggems_xray_source.argtypes = [
            ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float]
        ggems_lib.set_position_ggems_xray_source.restype = ctypes.c_void_p

        ggems_lib.print_infos_ggems_xray_source.argtypes = [ctypes.c_void_p]
        ggems_lib.print_infos_ggems_xray_source.restype = ctypes.c_void_p

        ggems_lib.set_source_particle_type_ggems_xray_source.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_source_particle_type_ggems_xray_source.restype =\
            ctypes.c_void_p

        ggems_lib.set_beam_aperture_ggems_xray_source.argtypes = [
            ctypes.c_void_p, ctypes.c_float]
        ggems_lib.set_beam_aperture_ggems_xray_source.restype = ctypes.c_void_p

        ggems_lib.set_focal_spot_size_ggems_xray_source.argtypes = [
            ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float]
        ggems_lib.set_focal_spot_size_ggems_xray_source.restype =\
            ctypes.c_void_p

        ggems_lib.set_local_axis_ggems_xray_source.argtypes = [
            ctypes.c_void_p,
            ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float, ctypes.c_float]
        ggems_lib.set_focal_spot_size_ggems_xray_source.restype =\
            ctypes.c_void_p

        ggems_lib.set_rotation_ggems_xray_source.argtypes = [
            ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float]
        ggems_lib.set_rotation_ggems_xray_source.restype = ctypes.c_void_p

        ggems_lib.update_rotation_ggems_xray_source.argtypes = [
            ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float]
        ggems_lib.update_rotation_ggems_xray_source.restype = ctypes.c_void_p

        ggems_lib.set_monoenergy_ggems_xray_source.argtypes = [
            ctypes.c_void_p, ctypes.c_float]
        ggems_lib.set_monoenergy_ggems_xray_source.restype = ctypes.c_void_p

        ggems_lib.set_polyenergy_ggems_xray_source.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_polyenergy_ggems_xray_source.restype = ctypes.c_void_p

        self.obj = ggems_lib.create_ggems_xray_source()

    def delete(self):
        ggems_lib.delete_ggems_xray_source(self.obj)

    def initialize(self):
        ggems_lib.initialize_ggems_xray_source(self.obj)

    def set_position(self, x, y, z):
        ggems_lib.set_position_ggems_xray_source(self.obj, x, y, z)

    def print_infos(self):
        ggems_lib.print_infos_ggems_xray_source(self.obj)

    def set_source_particle_type(self, particle_type):
        ggems_lib.set_source_particle_type_ggems_xray_source(
            self.obj, particle_type)

    def set_beam_aperture(self, beam_aperture):
        ggems_lib.set_beam_aperture_ggems_xray_source(self.obj, beam_aperture)

    def set_focal_spot_size(self, width, height, depth):
        ggems_lib.set_focal_spot_size_ggems_xray_source(
            self.obj, width, height, depth)

    def set_local_axis(self, m00, m01, m02, m10, m11, m12, m20, m21, m22):
        ggems_lib.set_local_axis_ggems_xray_source(
            self.obj, m00, m01, m02, m10, m11, m12, m20, m21, m22)

    def set_rotation(self, rx, ry, rz):
        ggems_lib.set_rotation_ggems_xray_source(self.obj, rx, ry, rz)

    def set_monoenergy(self, e):
        ggems_lib.set_monoenergy_ggems_xray_source(self.obj, e)

    def set_polyenergy(self, file):
        ggems_lib.set_polyenergy_ggems_xray_source(self.obj, file)

    def update_rotation(self, rx, ry, rz):
        ggems_lib.update_rotation_ggems_xray_source(self.obj, rx, ry, rz)


class GGEMSManager(object):
    """GGEMS class managing the simulation
    """
    def __init__(self):
        ggems_lib.get_instance_ggems_manager.restype = ctypes.c_void_p

        ggems_lib.set_seed_ggems_manager.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32]
        ggems_lib.set_seed_ggems_manager.restype = ctypes.c_void_p

        ggems_lib.initialize_ggems_manager.argtypes = [ctypes.c_void_p]
        ggems_lib.initialize_ggems_manager.restype = ctypes.c_void_p

        ggems_lib.set_number_of_particles_ggems_manager.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64]
        ggems_lib.set_number_of_particles_ggems_manager.restype =\
            ctypes.c_void_p

        ggems_lib.set_process_ggems_manager.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_process_ggems_manager.restype = ctypes.c_void_p

        ggems_lib.set_particle_cut_ggems_manager.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double]
        ggems_lib.set_particle_cut_ggems_manager.restype = ctypes.c_void_p

        ggems_lib.set_geometry_tolerance_ggems_manager.argtypes = [
            ctypes.c_void_p, ctypes.c_double]
        ggems_lib.set_geometry_tolerance_ggems_manager.restype =\
            ctypes.c_void_p

        ggems_lib.set_secondary_particle_and_level_ggems_manager.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
        ggems_lib.set_secondary_particle_and_level_ggems_manager.restype =\
            ctypes.c_void_p

        ggems_lib.set_cross_section_table_number_of_bins_ggems_manager\
            .argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        ggems_lib.set_cross_section_table_number_of_bins_ggems_manager.restype\
            = ctypes.c_void_p

        ggems_lib.set_cross_section_table_energy_min_ggems_manager.argtypes = [
            ctypes.c_void_p, ctypes.c_double]
        ggems_lib.set_cross_section_table_energy_min_ggems_manager.restype =\
            ctypes.c_void_p

        ggems_lib.set_cross_section_table_energy_max_ggems_manager.argtypes = [
            ctypes.c_void_p, ctypes.c_double]
        ggems_lib.set_cross_section_table_energy_max_ggems_manager.restype =\
            ctypes.c_void_p

        ggems_lib.run_ggems_manager.argtypes = [ctypes.c_void_p]
        ggems_lib.run_ggems_manager.restype = ctypes.c_void_p

        self.obj = ggems_lib.get_instance_ggems_manager()

    def set_seed(self, seed):
        ggems_lib.set_seed_ggems_manager(self.obj, seed)

    def initialize(self):
        ggems_lib.initialize_ggems_manager(self.obj)

    def set_number_of_particles(self, number_of_particles):
        ggems_lib.set_number_of_particles_ggems_manager(
            self.obj, number_of_particles)

    def set_process(self, process_name):
        ggems_lib.set_process_ggems_manager(self.obj, process_name)

    def set_particle_cut(self, particle_name, distance):
        ggems_lib.set_particle_cut_ggems_manager(
            self.obj, particle_name, distance)

    def set_geometry_tolerance(self, distance):
        ggems_lib.set_geometry_tolerance_ggems_manager(self.obj, distance)

    def set_secondary_particle_and_level(self, particle_name, level):
        ggems_lib.set_secondary_particle_and_level_ggems_manager(
            self.obj, particle_name, level)

    def set_cross_section_table_number_of_bins(self, number_of_bins):
        ggems_lib.set_cross_section_table_number_of_bins_ggems_manager(
            self.obj, number_of_bins)

    def set_cross_section_table_energy_min(self, min_energy):
        ggems_lib.set_cross_section_table_energy_min_ggems_manager(
            self.obj, min_energy)

    def set_cross_section_table_energy_max(self, max_energy):
        ggems_lib.set_cross_section_table_energy_max_ggems_manager(
            self.obj, max_energy)

    def run(self):
        ggems_lib.run_ggems_manager(self.obj)
