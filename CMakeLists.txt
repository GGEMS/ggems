#----------------------------------------------------------------------------
# Setup the project
#
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(GGEMS)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(STATUS "GGEMS requires an out-of-source build.")
    message(STATUS "Please remove these files from ${CMAKE_BINARY_DIR} first:")
    message(STATUS "CMakeCache.txt")
    message(STATUS "CMakeFiles")
    message(STATUS "Once these files are removed, create a separate directory")
    message(STATUS "and run CMake from there")
    message(FATAL_ERROR "in-source build detected")
endif()

# GEANT4
find_package(Geant4 REQUIRED)
include(${Geant4_USE_FILE})

# GGEMS src
include_directories(${PROJECT_SOURCE_DIR})
#file(GLOB sourcesGPU ${PROJECT_SOURCE_DIR}/*/*.cu)
#file(GLOB headersGPU ${PROJECT_SOURCE_DIR}/*/*.cuh)
file(GLOB_RECURSE sourcesGPU "*.cu")
file(GLOB_RECURSE headersGPU "*.cuh")

# Compilation
find_package(CUDA REQUIRED)
set(LIB_NAME "ggems")
set(LIBSCUDA --generate-code arch=compute_30,code=sm_30 --relocatable-device-code true -lcudadevrt --compiler-options '-fPIC' -use_fast_math -w )
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${LIBSCUDA})
set(CUDA_SEPARABLE_COMPILATION OFF)

cuda_add_library(${LIB_NAME} STATIC ${sourcesGPU})

# Installation
install(TARGETS ${LIB_NAME} DESTINATION "lib")
install(FILES ${headersGPU} DESTINATION "include")


