from ggems_lib import *

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