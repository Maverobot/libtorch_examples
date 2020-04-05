cmake_minimum_required(VERSION 3.4)
project(libtorch_examples)

set(LIBTORCH_PATH "" CACHE PATH "Path to the decompressed libtorch folder.")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_PATH})
find_package(Torch CONFIG REQUIRED)

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
