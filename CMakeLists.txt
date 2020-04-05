cmake_minimum_required(VERSION 3.4)
project(libtorch_examples)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(libtorch_VERSION 1.4.0)
find_package(Torch ${libtorch_VERSION} EXACT QUIET CONFIG)
if(NOT Torch_FOUND)
  message(STATUS "libtorch ${libtorch_VERSION} - not found")
  message(STATUS "Fetching libtorch")
  include(FetchContent)
  FetchContent_Declare(
    libtorch
    URL https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-${libtorch_VERSION}.zip
    SOURCE_DIR libtorch)
  FetchContent_GetProperties(libtorch)
  if(NOT libtorch_POPULATED)
    unset(FETCHCONTENT_QUIET CACHE)
    FetchContent_Populate(libtorch)
    list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR}/libtorch)
  endif()
  find_package(Torch ${libtorch_VERSION} EXACT CONFIG REQUIRED)
else()
  message(STATUS "libtorch ${libtorch_VERSION} - found")
endif()

macro(add_example name)
  add_executable(${name} src/${name}.cpp)
  target_include_directories(${name} PRIVATE include)
  target_link_libraries(${name} ${TORCH_LIBRARIES})
endmacro()

add_example(hello_world)
add_example(function_approx)
add_example(time_serie_prediction)
add_example(lstm_example)
add_example(dataset_example)

# Tools
include(${CMAKE_CURRENT_LIST_DIR}/cmake/ClangTools.cmake OPTIONAL
        RESULT_VARIABLE CLANG_TOOLS)
if(CLANG_TOOLS)
  file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
  file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
       ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h)
  add_format_target(${PROJECT_NAME} FILES ${SOURCES} ${HEADERS})
  add_tidy_target(${PROJECT_NAME} FILES ${SOURCES})
endif()
