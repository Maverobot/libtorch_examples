cmake_minimum_required(VERSION 3.4)
project(libtorch_examples)

set(LIBTORCH_PATH "" CACHE PATH "Path to the decompressed libtorch folder.")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_PATH})
find_package(Torch CONFIG REQUIRED)

add_executable(hello_world
  src/hello_world.cpp)
target_include_directories(hello_world PRIVATE
  include)
target_link_libraries(hello_world ${TORCH_LIBRARIES})

add_executable(function_approx
  src/function_approx.cpp)
target_include_directories(function_approx PRIVATE
  include)
target_link_libraries(function_approx ${TORCH_LIBRARIES})

add_executable(time_serie_prediction
  src/time_serie_prediction.cpp)
target_include_directories(time_serie_prediction PRIVATE
  include)
target_link_libraries(time_serie_prediction ${TORCH_LIBRARIES})

## Tools
include(${CMAKE_CURRENT_LIST_DIR}/cmake/ClangTools.cmake OPTIONAL
  RESULT_VARIABLE CLANG_TOOLS
)
if(CLANG_TOOLS)
  file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
  file(GLOB_RECURSE HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
  )
  add_format_target(${PROJECT_NAME} FILES ${SOURCES} ${HEADERS})
  add_tidy_target(${PROJECT_NAME}
    FILES ${SOURCES}
  )
endif()
