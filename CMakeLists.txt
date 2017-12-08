# cmake needs this line
cmake_minimum_required(VERSION 2.8)

# Define project name
project(trajectory_with_flow)

#########################################################
# FIND C++11 Support
#########################################################
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
####################################################################################################


# Find OpenCV and CERES
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
#find_package(gflags REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


# Add OpenCV headers location to your include paths
include_directories(${OpenCV_INCLUDE_DIRS}
                  ${CERES_INCLUDE_DIRS}
                  include)


add_library(helpers-cc src/triangulation.cc
                     src/sequential_probability_ratio.cc
                     src/random.cc
                     src/pose_util.cc
                     src/fundamental_matrix_util.cc
                     src/five_point.cpp
                     src/estimate_uncalibrated_relative_pose.cc
                     src/estimate_twoview_info.cc
                     src/estimate_relative_pose.cpp
                     src/essential_matrix_utils.cc
                     src/eight_point_fundamental_matrix.cc
                     src/correspondance.cc
                     src/helpers.cc)

add_library (dense-helpers-cc src/triangulate.cc
                            src/densehelpers.cc)

add_library (dense-optical_flow-tracker src/denseopticalflowtracker.cc)

add_library(sift-cc src/sift_processing.cc)

add_library(matrix src/matrix.cpp)


# Declare the executable target built from your sources
add_executable(trajectory_with_flow scripts/compute_trajactory.cpp)

# Link your application with OpenCV libraries
target_link_libraries(trajectory_with_flow ${OpenCV_LIBS}
                                           gflags
                                           glog
                                           tiff
                                           ${CERES_LIBRARIES}
                                           helpers-cc
                                           dense-optical_flow-tracker)



