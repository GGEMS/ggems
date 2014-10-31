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


find_package(Geant4 REQUIRED)
# find_package(Geant4Config REQUIRED)
set(BUILD_SHARED_LIBS OFF)
set(BUILD_STATIC_LIBS ON)
set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++ -std=c++98 -static")
# set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
option(STATICBUILD "Build library in static (otherwise in shared)" ON)

# Load some basic macros which are needed later on

# packages                                                                                                                                                                                                 
find_package(CUDA REQUIRED)
# SET(CUDA_NVCC_FLAGS "-arch=sm_30" CACHE STRING "nvcc flags" FORCE)
# nvcc flags                                      
# message("PROJECT_SOURCE_DIR points to " ${PROJECT_SOURCE_DIR})
# set(CUDA_NVCC_FLAGS arch=compute_30,code=sm_30 -Xptxas="-v" --relocatable-device-code true -lcudadevrt --compiler-options '-fPIC' --use_fast_math)
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_30,code=sm_30)
set(CUDA_SEPARABLE_COMPILATION OFF)

include(${Geant4_USE_FILE})

# add_library(Geant4LIB STATIC ${Geant4_INCLUDE_DIR})
# set_property(Geant4LIB PROPERTIES ${Geant4_LIBRARIES})
include_directories(${PROJECT_SOURCE_DIR})
    FILE(GLOB sourcesGPU
    ${PROJECT_SOURCE_DIR}/*.cu 
    )
    FILE(GLOB headersGPU
    ${PROJECT_SOURCE_DIR}/*.cuh 
    )
SET(GCC_COVERAGE_COMPILE_FLAGS "-w ")
#SET(GCC_COVERAGE_COMPILE_FLAGS "-w")
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

#----------------------------------------------------------------------------
# Add the executable, and link it to the Geant4 libraries
#
file(GLOB_RECURSE gpuacc_SRCS "*.cu")
# message("gpuacc_SRCS " ${gpuacc_SRCS})
# cuda_compile(BRACHYKERNEL gpubrachylib.cu)
set(LIB_NAME "GGEMS")
set(LIBSCUDA --generate-code arch=compute_30,code=sm_30 --relocatable-device-code true -lcudadevrt --compiler-options '-fPIC' -use_fast_math -w )
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${LIBSCUDA})
if(STATICBUILD)
cuda_add_library(${LIB_NAME} STATIC ${gpuacc_SRCS} )
else()
cuda_add_library(${LIB_NAME} SHARED ${gpuacc_SRCS} )
endif()
# cuda_add_library("GGEMS_static" STATIC ${gpuacc_SRCS} )
# add_library(GGEMSlib ${cuda_target})
# add_executable(brachy_ldr brachy_ldr.cpp ${sources} ${headers} ${sourcesutils} ${headersutils} ${BRACHYKERNEL})
# target_link_libraries(brachy_ldr ${Geant4_LIBRARIES})
# set_property(brachy_ldr ${Geant4LIB} ${Geant4_LIBRARIES} *.a)
# add_library(Geant4lib ${Geant4_LIBRARIES})

#----------------------------------------------------------------------------
# For internal Geant4 use - but has no effect if you build this
# example standalone
#
# add_custom_target(MAIN DEPENDS brachy_ldr)

#----------------------------------------------------------------------------
# Install the executable to 'bin' directory under CMAKE_INSTALL_PREFIX
#
install(TARGETS ${LIB_NAME} DESTINATION lib)
# install(PATTERN "*.cuh" DESTINATION include)
# install(LIBRARY DESTINATION lib)
# install(TARGETS ${headersGPU} DESTINATION include)


