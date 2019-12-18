import os
import shutil
import sys

# ------------------------------------------------------------------------------
# Choose your compiler: 'CLANG', 'GCC', 'CL' (Visual Studio) depending on your
# OS
if sys.platform == "linux" or sys.platform == "darwin":
    os.environ['COMPILER'] = 'CLANG'
    os.environ['CC'] = 'clang-9'
    os.environ['CXX'] = 'clang++-9'
elif sys.platform == "win32":
    os.environ['COMPILER'] = 'CL'
    os.environ['CC'] = 'cl.exe'
    os.environ['CXX'] = 'cl.exe'
else:  # Unknown system
    print("Unknown architecture!!!", file=sys.stderr)

# ------------------------------------------------------------------------------
# Set the GGEMS folder (GGEMS source folder), the build folder (where GGEMS
# will be compiled) and the install folder
if sys.platform == "linux" or sys.platform == "darwin":
    GGEMS_FOLDER = "/home/dbenoit/Desktop/GGEMS"
    BUILD_FOLDER = "/home/dbenoit/data/Build/GGEMS_OpenCL"
    INSTALL_FOLDER = "/home/dbenoit"
elif sys.platform == "win32":
    GGEMS_FOLDER = "C:\\Users\\dbenoit\\Workspace\\GGEMS_OpenCL"
    BUILD_FOLDER = "C:\\Users\\dbenoit\\Workspace\\GGEMS_OpenCL_build"
    INSTALL_FOLDER = "C:\\Users\\dbenoit"

# ------------------------------------------------------------------------------
# Print infos
print('')
print('***********')
print('Compiler:', os.environ['COMPILER'])
print('GGEMS folder:', GGEMS_FOLDER)
print('GGEMS build folder:', BUILD_FOLDER)
print('GGEMS install folder:', INSTALL_FOLDER)
print('***********')
print('')

# ------------------------------------------------------------------------------
# Delete CMAKE cache and file
if os.path.exists(BUILD_FOLDER + "/CMakeCache.txt"):
    print('Removing CMAKE cache...')
    os.remove(BUILD_FOLDER + "/CMakeCache.txt")

if os.path.isdir(BUILD_FOLDER + "/CMakeFiles"):
    print('Removing CMakeFiles...')
    shutil.rmtree(BUILD_FOLDER + "/CMakeFiles")

# ------------------------------------------------------------------------------
# Launching CMAKE
cmake_cmd = "cmake"
if sys.platform == "linux" or sys.platform == "darwin":
    cmake_cmd += " -G \"Unix Makefiles\""
elif sys.platform == "win32":
    cmake_cmd += " -G \"Ninja\""
cmake_cmd += " -DCMAKE_BUILD_TYPE=Release"
cmake_cmd += " -DOPENCL_KERNEL_PATH=" + GGEMS_FOLDER + "/src/kernels"
cmake_cmd += " -DGGEMS_PATH=" + GGEMS_FOLDER
cmake_cmd += " -DMAXIMUM_PARTICLES=100000"
cmake_cmd += " -DCMAKE_VERBOSE_MAKEFILE=OFF"
cmake_cmd += " -DCOMPILER=" + os.environ['COMPILER']
cmake_cmd += " -DCMAKE_INSTALL_PREFIX=" + INSTALL_FOLDER
cmake_cmd += " -S " + GGEMS_FOLDER + " -B " + BUILD_FOLDER

os.system(cmake_cmd)
