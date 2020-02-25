import ctypes
import sys

if sys.platform == "linux":
    ggems_lib = ctypes.cdll.LoadLibrary("/home/dbenoit/data/Build/GGEMS_OpenCL/libggems.so")
elif sys.platform == "darwin":
    ggems_lib = ctypes.cdll.LoadLibrary("/home/dbenoit/data/Build/GGEMS_OpenCL/libggems.dylib")
elif sys.platform == "win32":
    ggems_lib = ctypes.cdll.LoadLibrary("C:\\Users\\dbenoit\\Workspace\\GGEMS_OpenCL_build\\libggems.dll")


class GGEMSOpenCLManager(object):
    """Get the OpenCL C++ singleton and print infos or managing it
    """
    def __init__(self):
        ggems_lib.get_instance_ggems_opencl_manager.restype = ctypes.c_void_p

        ggems_lib.print_infos_opencl_manager.argtypes = [ctypes.c_void_p]
        ggems_lib.print_infos_opencl_manager.restype = ctypes.c_void_p

        ggems_lib.clean_opencl_manager.argtypes = [ctypes.c_void_p]
        ggems_lib.clean_opencl_manager.restype = ctypes.c_void_p

        ggems_lib.print_RAM_ggems_opencl_manager.argtypes = [ctypes.c_void_p]
        ggems_lib.print_RAM_ggems_opencl_manager.restype = ctypes.c_void_p

        ggems_lib.set_context_index_ggems_opencl_manager.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        ggems_lib.set_context_index_ggems_opencl_manager.restype = ctypes.c_void_p

        self.obj = ggems_lib.get_instance_ggems_opencl_manager()

    def print_infos(self):
        ggems_lib.print_infos_opencl_manager(self.obj)

    def print_RAM(self):
        ggems_lib.print_RAM_ggems_opencl_manager(self.obj)

    def set_context_index(self, context_id):
        ggems_lib.set_context_index_ggems_opencl_manager(self.obj, context_id)

    def clean(self):
        ggems_lib.clean_opencl_manager(self.obj)


class GGEMSVerbosity(object):
    """Set the verbosity of infos in GGEMS
    """
    def __init__(self, val):
        ggems_lib.set_ggems_verbose.argtypes = [ctypes.c_int]
        ggems_lib.set_ggems_verbose.restype = ctypes.c_void_p

        ggems_lib.set_ggems_verbose(val)


class GGEMSPhantomNavigatorManager(object):
    """Class managing phantom navigator in GGEMS
    """
    def __init__(self):
        ggems_lib.get_instance_ggems_phantom_navigator_manager.restype = ctypes.c_void_p

        ggems_lib.print_infos_ggems_phantom_navigator_manager.argtypes = [ctypes.c_void_p]
        ggems_lib.print_infos_ggems_phantom_navigator_manager.restype = ctypes.c_void_p

        self.obj = ggems_lib.get_instance_ggems_phantom_navigator_manager()

    def print_infos(self):
        ggems_lib.print_infos_ggems_phantom_navigator_manager(self.obj)


class GGEMSVoxelizedPhantomNavigatorImagery(object):
    """Class for the voxelized phantom navigator for imagery application
    """
    def __init__(self):
        ggems_lib.create_ggems_voxelized_phantom_navigator_imagery.restype = ctypes.c_void_p

        ggems_lib.set_phantom_name_ggems_voxelized_phantom_navigator_imagery.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_phantom_name_ggems_voxelized_phantom_navigator_imagery.restype = ctypes.c_void_p

        ggems_lib.set_phantom_file_ggems_voxelized_phantom_navigator_imagery.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_phantom_file_ggems_voxelized_phantom_navigator_imagery.restype = ctypes.c_void_p

        ggems_lib.set_range_to_material_filename_ggems_voxelized_phantom_navigator_imagery.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_range_to_material_filename_ggems_voxelized_phantom_navigator_imagery.restype = ctypes.c_void_p

        ggems_lib.set_geometry_tolerance_ggems_voxelized_phantom_navigator_imagery.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_char_p]
        ggems_lib.set_geometry_tolerance_ggems_voxelized_phantom_navigator_imagery.restype = ctypes.c_void_p

        self.obj = ggems_lib.create_ggems_voxelized_phantom_navigator_imagery()

    def set_phantom_name(self, name):
        ggems_lib.set_phantom_name_ggems_voxelized_phantom_navigator_imagery(self.obj, name)

    def set_phantom_image(self, name):
        ggems_lib.set_phantom_file_ggems_voxelized_phantom_navigator_imagery(self.obj, name)

    def set_range_to_material(self, name):
        ggems_lib.set_range_to_material_filename_ggems_voxelized_phantom_navigator_imagery(self.obj, name)

    def set_geometry_tolerance(self, distance, unit):
        ggems_lib.set_geometry_tolerance_ggems_voxelized_phantom_navigator_imagery(self.obj, distance, unit)


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
        ggems_lib.set_source_name_ggems_xray_source(self.obj, name)

    def set_position(self, x, y, z, unit):
        ggems_lib.set_position_ggems_xray_source(self.obj, x, y, z, unit)

    def set_number_of_particles(self, number_of_particles):
        ggems_lib.set_number_of_particles_xray_source(self.obj, number_of_particles)

    def set_source_particle_type(self, particle_type):
        ggems_lib.set_source_particle_type_ggems_xray_source(self.obj, particle_type)

    def set_beam_aperture(self, beam_aperture, unit):
        ggems_lib.set_beam_aperture_ggems_xray_source(self.obj, beam_aperture, unit)

    def set_focal_spot_size(self, width, height, depth, unit):
        ggems_lib.set_focal_spot_size_ggems_xray_source(self.obj, width, height, depth, unit)

    def set_local_axis(self, m00, m01, m02, m10, m11, m12, m20, m21, m22):
        ggems_lib.set_local_axis_ggems_xray_source(self.obj, m00, m01, m02, m10, m11, m12, m20, m21, m22)

    def set_rotation(self, rx, ry, rz, unit):
        ggems_lib.set_rotation_ggems_xray_source(self.obj, rx, ry, rz, unit)

    def set_monoenergy(self, e, unit):
        ggems_lib.set_monoenergy_ggems_xray_source(self.obj, e, unit)

    def set_polyenergy(self, file):
        ggems_lib.set_polyenergy_ggems_xray_source(self.obj, file)


class GGEMSMaterialsManager(object):
    """Class handling the materials in GGEMS
    """
    def __init__(self):
        ggems_lib.get_instance_materials_manager.restype = ctypes.c_void_p

        ggems_lib.set_materials_ggems_materials_manager.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_materials_ggems_materials_manager.restype = ctypes.c_void_p

        ggems_lib.print_available_chemical_elements_ggems_materials_manager.argtypes = [ctypes.c_void_p]
        ggems_lib.print_available_chemical_elements_ggems_materials_manager.restype = ctypes.c_void_p

        ggems_lib.print_available_materials_ggems_materials_manager.argtypes = [ctypes.c_void_p]
        ggems_lib.print_available_materials_ggems_materials_manager.restype = ctypes.c_void_p

        self.obj = ggems_lib.get_instance_materials_manager()

    def set_materials(self, filename):
        ggems_lib.set_materials_ggems_materials_manager(self.obj, filename)

    def print_available_chemical_elements(self):
        ggems_lib.print_available_chemical_elements_ggems_materials_manager(self.obj)

    def print_available_materials(self):
        ggems_lib.print_available_materials_ggems_materials_manager(self.obj)


class GGEMSManager(object):
    """GGEMS class managing the simulation
    """
    def __init__(self):
        ggems_lib.get_instance_ggems_manager.restype = ctypes.c_void_p

        ggems_lib.set_seed_ggems_manager.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        ggems_lib.set_seed_ggems_manager.restype = ctypes.c_void_p

        ggems_lib.initialize_ggems_manager.argtypes = [ctypes.c_void_p]
        ggems_lib.initialize_ggems_manager.restype = ctypes.c_void_p

        ggems_lib.set_process_ggems_manager.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_process_ggems_manager.restype = ctypes.c_void_p

        ggems_lib.set_particle_cut_ggems_manager.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double]
        ggems_lib.set_particle_cut_ggems_manager.restype = ctypes.c_void_p

        ggems_lib.set_secondary_particle_and_level_ggems_manager.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
        ggems_lib.set_secondary_particle_and_level_ggems_manager.restype = ctypes.c_void_p

        ggems_lib.set_cross_section_table_number_of_bins_ggems_manager.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        ggems_lib.set_cross_section_table_number_of_bins_ggems_manager.restype = ctypes.c_void_p

        ggems_lib.set_cross_section_table_energy_min_ggems_manager.argtypes = [ctypes.c_void_p, ctypes.c_double]
        ggems_lib.set_cross_section_table_energy_min_ggems_manager.restype = ctypes.c_void_p

        ggems_lib.set_cross_section_table_energy_max_ggems_manager.argtypes = [ctypes.c_void_p, ctypes.c_double]
        ggems_lib.set_cross_section_table_energy_max_ggems_manager.restype = ctypes.c_void_p

        ggems_lib.run_ggems_manager.argtypes = [ctypes.c_void_p]
        ggems_lib.run_ggems_manager.restype = ctypes.c_void_p

        self.obj = ggems_lib.get_instance_ggems_manager()

    def set_seed(self, seed):
        ggems_lib.set_seed_ggems_manager(self.obj, seed)

    def initialize(self):
        ggems_lib.initialize_ggems_manager(self.obj)

    def set_process(self, process_name):
        ggems_lib.set_process_ggems_manager(self.obj, process_name)

    def set_particle_cut(self, particle_name, distance):
        ggems_lib.set_particle_cut_ggems_manager(self.obj, particle_name, distance)

    def set_secondary_particle_and_level(self, particle_name, level):
        ggems_lib.set_secondary_particle_and_level_ggems_manager(self.obj, particle_name, level)

    def set_cross_section_table_number_of_bins(self, number_of_bins):
        ggems_lib.set_cross_section_table_number_of_bins_ggems_manager(self.obj, number_of_bins)

    def set_cross_section_table_energy_min(self, min_energy):
        ggems_lib.set_cross_section_table_energy_min_ggems_manager(self.obj, min_energy)

    def set_cross_section_table_energy_max(self, max_energy):
        ggems_lib.set_cross_section_table_energy_max_ggems_manager(self.obj, max_energy)

    def run(self):
        ggems_lib.run_ggems_manager(self.obj)


class GGEMSPhantomCreatorManager(object):
    """Get Phantom Creator Manager to convert analytical volume to voxelized
    volume
    """
    def __init__(self):
        ggems_lib.get_instance_phantom_creator_manager.restype = ctypes.c_void_p

        ggems_lib.set_phantom_dimension_phantom_creator_manager.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        ggems_lib.set_phantom_dimension_phantom_creator_manager.restype = ctypes.c_void_p

        ggems_lib.set_element_sizes_phantom_creator_manager.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_char_p]
        ggems_lib.set_element_sizes_phantom_creator_manager.restype = ctypes.c_void_p

        ggems_lib.set_output_image_filename_phantom_creator_manager.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_output_image_filename_phantom_creator_manager.restype = ctypes.c_void_p

        ggems_lib.set_output_range_to_material_filename_phantom_creator_manager.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_output_range_to_material_filename_phantom_creator_manager.restype = ctypes.c_void_p

        ggems_lib.initialize_phantom_creator_manager.argtypes = [ctypes.c_void_p]
        ggems_lib.initialize_phantom_creator_manager.restype = ctypes.c_void_p

        ggems_lib.write_phantom_creator_manager.argtypes = [ctypes.c_void_p]
        ggems_lib.write_phantom_creator_manager.restype = ctypes.c_void_p

        ggems_lib.set_isocenter_positions.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_char_p]
        ggems_lib.set_isocenter_positions.restype = ctypes.c_void_p

        ggems_lib.set_material_phantom_creator_manager.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_material_phantom_creator_manager.restype = ctypes.c_void_p

        self.obj = ggems_lib.get_instance_phantom_creator_manager()

    def set_dimensions(self, width, height, depth):
        ggems_lib.set_phantom_dimension_phantom_creator_manager(self.obj, width, height, depth)

    def set_element_sizes(self, width, height, depth, unit):
        ggems_lib.set_element_sizes_phantom_creator_manager(self.obj, width, height, depth, unit)

    def set_output(self, output):
        ggems_lib.set_output_image_filename_phantom_creator_manager(self.obj, output)

    def set_range_output(self, output):
        ggems_lib.set_output_range_to_material_filename_phantom_creator_manager(self.obj, output)

    def initialize(self):
        ggems_lib.initialize_phantom_creator_manager(self.obj)

    def write(self):
        ggems_lib.write_phantom_creator_manager(self.obj)

    def set_isocenter_positions(self, iso_pos_x, iso_pos_y, iso_pos_z, unit):
        ggems_lib.set_isocenter_positions(self.obj, iso_pos_x, iso_pos_y, iso_pos_z, unit)

    def set_material(self, material):
        ggems_lib.set_material_phantom_creator_manager(self.obj, material)


class GGEMSTube(object):
    """Build a solid tube analytical phantom
    """
    def __init__(self):
        ggems_lib.create_tube.restype = ctypes.c_void_p

        ggems_lib.delete_tube.argtypes = [ctypes.c_void_p]
        ggems_lib.delete_tube.restype = ctypes.c_void_p

        ggems_lib.set_height_tube.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_char_p]
        ggems_lib.set_height_tube.restype = ctypes.c_void_p

        ggems_lib.set_radius_tube.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_char_p]
        ggems_lib.set_radius_tube.restype = ctypes.c_void_p

        ggems_lib.set_position_tube.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_char_p]
        ggems_lib.set_position_tube.restype = ctypes.c_void_p

        ggems_lib.set_label_value_tube.argtypes = [ctypes.c_void_p, ctypes.c_float]
        ggems_lib.set_label_value_tube.restype = ctypes.c_void_p

        ggems_lib.set_material_tube.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        ggems_lib.set_material_tube.restype = ctypes.c_void_p

        ggems_lib.initialize_tube.argtypes = [ctypes.c_void_p]
        ggems_lib.initialize_tube.restype = ctypes.c_void_p

        ggems_lib.draw_tube.argtypes = [ctypes.c_void_p]
        ggems_lib.draw_tube.restype = ctypes.c_void_p

        self.obj = ggems_lib.create_tube()

    def delete(self):
        ggems_lib.delete_tube(self.obj)

    def set_height(self, height, unit):
        ggems_lib.set_height_tube(self.obj, height, unit)

    def set_radius(self, radius, unit):
        ggems_lib.set_radius_tube(self.obj, radius, unit)

    def set_label_value(self, label_value):
        ggems_lib.set_label_value_tube(self.obj, label_value)

    def set_position(self, pos_x, pos_y, pos_z, unit):
        ggems_lib.set_position_tube(self.obj, pos_x, pos_y, pos_z, unit)

    def set_material(self, material):
        ggems_lib.set_material_tube(self.obj, material)

    def initialize(self):
        ggems_lib.initialize_tube(self.obj)

    def draw(self):
        ggems_lib.draw_tube(self.obj)


# ------------------------------------------------------------------------------
# Setting global verbosity to 0 for initialization
# 0 - minimum infos
# 3 - max infos, maybe too much!!!
GGEMSVerbosity(3)

# ------------------------------------------------------------------------------
# Calling all C++ singleton managers
opencl_manager = GGEMSOpenCLManager()
material_manager = GGEMSMaterialsManager()
phantom_manager = GGEMSPhantomNavigatorManager()
source_manager = GGEMSSourceManager()
ggems_manager = GGEMSManager()
phantom_creator_manager = GGEMSPhantomCreatorManager()
