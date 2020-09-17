import os
import shutil
import sys
import glob
import pathlib

# ------------------------------------------------------------------------------
# Choose your compiler: 'CLANG', 'GCC', 'CL' (Visual Studio) depending on your OS
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
# Check if option 'clean' is given
is_clean = False
if len(sys.argv) > 1:
    if sys.argv[1] != 'clean':
        print('Error: argument has to be \'clean\'!!!')
        sys.exit(2)
    else:
        is_clean = True

# ------------------------------------------------------------------------------
# Set the GGEMS folder (GGEMS source folder), the build folder (where GGEMS will be compiled) and the install folder
GGEMS_FOLDER = os.path.abspath(os.path.dirname(sys.argv[0]))
BUILD_FOLDER = os.path.join(os.path.dirname(GGEMS_FOLDER),"GGEMS_OpenCL_build")
INSTALL_FOLDER = os.path.expanduser("~")

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
# Delete CMAKE cache and file if 'clean' option
if is_clean:
    if os.path.exists(BUILD_FOLDER + "/CMakeCache.txt"):
        print('Removing CMAKE cache...')
        os.remove(BUILD_FOLDER + "/CMakeCache.txt")

    if os.path.isdir(BUILD_FOLDER + "/CMakeFiles"):
        print('Removing CMakeFiles...')
        shutil.rmtree(BUILD_FOLDER + "/CMakeFiles")

    # Remove all files
    files = pathlib.Path(BUILD_FOLDER).glob("*.*")
    for p in files:
        p.unlink()

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
cmake_cmd += " -DMAXIMUM_PARTICLES=1048576"
cmake_cmd += " -DCMAKE_VERBOSE_MAKEFILE=OFF"
cmake_cmd += " -DOPENCL_CACHE_KERNEL_COMPILATION=OFF"
cmake_cmd += " -DCOMPILER=" + os.environ['COMPILER']
cmake_cmd += " -DCMAKE_INSTALL_PREFIX=" + INSTALL_FOLDER
cmake_cmd += " -S " + GGEMS_FOLDER + " -B " + BUILD_FOLDER

os.system(cmake_cmd)
