# Require at least version 2.8 of CMake
cmake_minimum_required(VERSION 2.8)

set(PROJ_NAME mnist_arma_blas)

# The project name. A project can have multiple targets
project(${PROJ_NAME})

include_directories(
  "/usr/local/include"
  ${CMAKE_BINARY_DIR}
)

find_package(Freetype REQUIRED)

add_definitions("-DDATA_DIR=${CMAKE_SOURCE_DIR}")
add_definitions("-DNO_FREETYPE")

configure_file(config.h.in
  config.h
)

find_library(PNGWRITER_LIB pngwriter
  /usr/local/lib
)

set(SOURCE_FILES
  main.cpp
  util.cpp
)

set(HEADER_FILES
  fc.hpp
  gemm.hpp
  mat.hpp
  mnist.hpp
  neural_net.hpp
  png_weight.hpp
  util.hpp
)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)

if(COMPILER_SUPPORTS_CXX11)
   list(APPEND CMAKE_CXX_FLAGS -std=c++11)
else()
   message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# Add the executable target. It does not need to
# have the same name as the project
add_executable(${PROJ_NAME}
  ${SOURCE_FILES}
  ${HEADER_FILES}
)

target_include_directories(${PROJ_NAME} PRIVATE
  "${CMAKE_BUILD_DIR}"
)

target_link_libraries(${PROJ_NAME}
  "-framework Accelerate"
  ${PNGWRITER_LIB}
)
