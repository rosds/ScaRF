cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(ScaRF)

##
# 2014-09-18
#
#   ScaRF Project
#
#   ScaRF is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   ScaRF is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with ScaRF.  If not, see <http://www.gnu.org/licenses/>.
#
##

set(CMAKE_BUILD_TYPE DEBUG)

# Find packages
find_package(Doxygen)
find_package(CUDA REQUIRED)
find_package(Boost REQUIRED)


# Include third-party include files
include_directories(${Boost_INCLUDE_DIRS}) 


set(SCARF_SRC 
    include/scarf/DataSet.cuh
    include/scarf/RFConfig.cuh
    include/scarf/RF.cuh
    include/scarf/InformationGain.cuh
    include/scarf/Tree/Tree.cuh
    include/scarf/Tree/TreeTrainingUtils.cuh
)


if(CMAKE_BUILD_TYPE MATCHES DEBUG)
    set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -arch=sm_20 --ptxas-options=-v)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG) 


cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
cuda_add_library(scarf SHARED ${SCARF_SRC})


# Set output directory for executables and libraries
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)


# Generate samples with 'make samples'
add_subdirectory(samples)
add_custom_target(samples
    DEPENDS scarf simple_example
)

add_definitions("-Wall -Wextra -g")

# Generate Documentation with 'make doc'
if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(doc
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen" VERBATIM
    )
endif(DOXYGEN_FOUND)
