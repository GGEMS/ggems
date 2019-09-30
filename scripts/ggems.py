import ctypes

# The user has to set the location of the libggems.so/dll/dylib library
ggems_lib = ctypes.cdll.LoadLibrary(
  "/home/dbenoit/data/Build/GGEMS_OpenCL/libggems.so")


class OpenCLManager(object):
    """Get the OpenCL C++ singleton and print infos or managing it
    """
    def __init__(self):
        ggems_lib.get_instance.restype = ctypes.c_void_p

        ggems_lib.print_platform.argtypes = [ctypes.c_void_p]
        ggems_lib.print_platform.restype = ctypes.c_void_p

        ggems_lib.print_device.argtypes = [ctypes.c_void_p]
        ggems_lib.print_device.restype = ctypes.c_void_p

        ggems_lib.print_build_options.argtypes = [ctypes.c_void_p]
        ggems_lib.print_build_options.restype = ctypes.c_void_p

        ggems_lib.print_context.argtypes = [ctypes.c_void_p]
        ggems_lib.print_context.restype = ctypes.c_void_p

        ggems_lib.print_RAM.argtypes = [ctypes.c_void_p]
        ggems_lib.print_RAM.restype = ctypes.c_void_p

        ggems_lib.print_command_queue.argtypes = [ctypes.c_void_p]
        ggems_lib.print_command_queue.restype = ctypes.c_void_p

        ggems_lib.set_context_index.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32]
        ggems_lib.set_context_index.restype = ctypes.c_void_p

        ggems_lib.print_activated_context.argtypes = [ctypes.c_void_p]
        ggems_lib.print_activated_context.restype = ctypes.c_void_p

        self.obj = ggems_lib.get_instance()

    def print_info(self):
        ggems_lib.print_platform(self.obj)

    def print_device(self):
        ggems_lib.print_device(self.obj)

    def print_build_options(self):
        ggems_lib.print_build_options(self.obj)

    def print_context(self):
        ggems_lib.print_context(self.obj)

    def print_RAM(self):
        ggems_lib.print_RAM(self.obj)

    def print_command_queue(self):
        ggems_lib.print_command_queue(self.obj)

    def set_context_index(self, context_id):
        ggems_lib.set_context_index(self.obj, context_id)

    def print_activated_context(self):
        ggems_lib.print_activated_context(self.obj)


class Verbosity(object):
    """Set the verbosity of infos in GGEMS
    """
    def __init__(self, val):
        ggems_lib.set_verbose.argtypes = [ctypes.c_int]
        ggems_lib.set_verbose.restype = ctypes.c_void_p

        ggems_lib.set_verbose(val)

#
#class GGEMS(object):
#    """GGEMS class managing the simulation
#    """
#    def __init__(self):
