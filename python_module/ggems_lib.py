import ctypes
import sys

# ------------------------------------------------------------------------------
# Get the location of ggems library and set verbosity. This file could be
# created during installation, transparent for user

if sys.platform == "linux":
    ggems_lib = ctypes.cdll.LoadLibrary("/home/dbenoit/data/Build/GGEMS_OpenCL/libggems.so")
elif sys.platform == "darwin":
    ggems_lib = ctypes.cdll.LoadLibrary("/home/dbenoit/data/Build/GGEMS_OpenCL/libggems.dylib")
elif sys.platform == "win32":
    ggems_lib = ctypes.cdll.LoadLibrary("C:\\Users\\dbenoit\\Workspace\\GGEMS_OpenCL_build\\libggems.dll")

class GGEMSVerbosity(object):
    """Set the verbosity of infos in GGEMS
    """
    def __init__(self, val):
        ggems_lib.set_ggems_verbose.argtypes = [ctypes.c_int]
        ggems_lib.set_ggems_verbose.restype = ctypes.c_void_p

        ggems_lib.set_ggems_verbose(val)


# ------------------------------------------------------------------------------
# Setting global verbosity to 0 for initialization
# 0 - minimum infos
# 3 - max infos, maybe too much!!!
GGEMSVerbosity(3)
