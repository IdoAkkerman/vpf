# This file is part of the VPF application.
#
# VPF is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.
#-------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.8)
project(VPF LANGUAGES CXX C)

#-------------------------------------------------------------------------------
# Check MFEM_DIR
#-------------------------------------------------------------------------------
if(NOT DEFINED MFEM_DIR)
  message(FATAL_ERROR "MFEM_DIR not defined")
endif()

message(STATUS "MFEM_DIR = ${MFEM_DIR}")
if(NOT IS_DIRECTORY ${MFEM_DIR})
  message(FATAL_ERROR "MFEM_DIR is not a directory")
endif()

#-------------------------------------------------------------------------------
# Add Files
#-------------------------------------------------------------------------------
set(SOURCES
    printInfo.cpp)

set(HEADERS
    buildInfo.hpp)

#-------------------------------------------------------------------------------
# Create header
#-------------------------------------------------------------------------------
get_filename_component(tmp1 ${MFEM_DIR} REALPATH)
get_filename_component(tmp2 ${PROJECT_BINARY_DIR}/.. REALPATH)

if (${tmp1} STREQUAL ${tmp2})
  message(STATUS "Report VPF, MFEM and hypre in build info.")
  add_custom_command(OUTPUT buildInfo.hpp
                     PRE_BUILD
                     COMMAND /bin/bash
                             "${PROJECT_SOURCE_DIR}/../config/buildInfo.sh"
                             "${PROJECT_SOURCE_DIR}" "mfem" "hypre"
                     COMMENT "Getting git info")
else()
  message(STATUS "External MFEM and hypre libraries used.")
  message(STATUS "Only report VPF in build info.")
  add_custom_command(OUTPUT buildInfo.hpp
                     PRE_BUILD
                     COMMAND /bin/bash
                             "${PROJECT_SOURCE_DIR}/../config/buildInfo.sh"
                             "${PROJECT_SOURCE_DIR}"
                     COMMENT "Getting git info")
endif()

#-------------------------------------------------------------------------------
# Add packages
#-------------------------------------------------------------------------------
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../config)

#-------------------------------------------------------------------------------
# Add MPI
#-------------------------------------------------------------------------------
find_package(MPI REQUIRED)

set(MPICXX mpicxx)
set(MPI_CXX_INCLUDE_DIRS ${MPI_CXX_INCLUDE_PATH})
set(CMAKE_CXX_COMPILER ${MPICXX})

#-------------------------------------------------------------------------------
# Add mfem
#-------------------------------------------------------------------------------
find_package(mfem REQUIRED NAMES MFEM HINTS "${MFEM_DIR}")

message(STATUS "Found mfem: in ${MFEM_DIR} (version ${MFEM_VERSION})")
message(STATUS "MFEM_CXX_FLAGS = ${MFEM_CXX_FLAGS}")
message(STATUS "MFEM_INCLUDE_DIRS = ${MFEM_INCLUDE_DIRS}")
message(STATUS "MFEM_LIBRARIES = ${MFEM_LIBRARIES}")

include_directories(${MFEM_INCLUDE_DIRS})

#-------------------------------------------------------------------------------
# Build executable
#-------------------------------------------------------------------------------
add_executable(vpf vpf.cpp ${SOURCES} ${HEADERS})

target_link_libraries(vpf PRIVATE ${MFEM_LIBRARIES})
install(TARGETS vpf DESTINATION bin)

