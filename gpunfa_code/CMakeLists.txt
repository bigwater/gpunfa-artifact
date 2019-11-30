cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
enable_language(CXX CUDA)

project(GPUNFA2019)
find_package(CUDA REQUIRED)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -lineinfo --expt-extended-lambda -O3 -Wno-deprecated-gpu-targets")

include_directories(.)
include_directories(include)
include_directories(${CUDA_INCLUDE_DIRS})

file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_SOURCE_DIR} "src/commons/*.cpp")
add_library(gpunfacommons ${SOURCES})
target_include_directories(gpunfacommons PRIVATE include/commons include/pugixml )


file(GLOB_RECURSE SOURCES1 RELATIVE ${CMAKE_SOURCE_DIR} "src/gpunfautils/*")
cuda_add_library(gpunfautils ${SOURCES1})
target_include_directories(gpunfautils PRIVATE include/commons include/gpunfautils)
set_property(TARGET gpunfautils PROPERTY CUDA_STANDARD 11)


# CUDA Projects
# obat
SET(PROJ "obat")
file(GLOB_RECURSE SOURCES2 RELATIVE ${CMAKE_SOURCE_DIR} "src/${PROJ}/*")
cuda_add_executable(${PROJ} ${SOURCES2})
set_property(TARGET ${PROJ} PROPERTY CUDA_STANDARD 11)
target_link_libraries(${PROJ} gpunfacommons gpunfautils)
target_include_directories(${PROJ} PRIVATE include/commons;include/gpunfautils)

# infant
file(GLOB_RECURSE SOURCES_INFANT RELATIVE ${CMAKE_SOURCE_DIR} "src/infant/*")
cuda_add_executable(infant ${SOURCES_INFANT})
set_property(TARGET infant PROPERTY CUDA_STANDARD 11)
target_link_libraries(infant gpunfacommons gpunfautils)

# ppopp12
file(GLOB_RECURSE SOURCES_PPOPP RELATIVE ${CMAKE_SOURCE_DIR} "src/ppopp12/*")
cuda_add_executable(ppopp12 ${SOURCES_PPOPP})
set_property(TARGET ppopp12 PROPERTY CUDA_STANDARD 11)
target_link_libraries(ppopp12 gpunfacommons gpunfautils)
target_include_directories(ppopp12 PRIVATE include/commons;include/gpunfautils)




