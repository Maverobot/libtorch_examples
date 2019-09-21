include(CMakeParseArguments)

find_program(CLANG_FORMAT_PROG clang-format DOC "'clang-format' executable")
if(CLANG_FORMAT_PROG AND NOT TARGET format)
  add_custom_target(format)
  add_custom_target(check-format)
endif()

find_program(CLANG_TIDY_PROG clang-tidy DOC "'clang-tidy' executable")
if(CLANG_TIDY_PROG AND NOT TARGET tidy)
  if(NOT CMAKE_EXPORT_COMPILE_COMMANDS)
    message(WARNING "Invoke Catkin/CMake with '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'
                     to generate compilation database for 'clang-tidy'.")
  endif()
  add_custom_target(tidy)
  add_custom_target(check-tidy)
endif()

find_program(PARALLEL_PROG parallel DOC "'parallel' executable")
if(PARALLEL_PROG)
  set(PARALLEL_DELIMITER ":::")
else()
  message(WARNING "parallel is not installeld. To enable parallelization of clang-tidy, try with 'sudo apt install parallel'.")
  set(PARALLEL_PROG "")
  set(PARALLEL_DELIMITER "")
endif()

function(add_format_target _target)
  if(NOT CLANG_FORMAT_PROG)
    return()
  endif()
  cmake_parse_arguments(ARG "" "" "FILES" ${ARGN})

  add_custom_target(format-${_target}
    COMMAND ${CLANG_FORMAT_PROG} -i ${ARG_FILES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
    COMMENT "Formatting ${_target} source code with clang-format"
    VERBATIM
  )
  add_dependencies(format format-${_target})

  add_custom_target(check-format-${_target}
    COMMAND ${CLANG_FORMAT_PROG} -output-replacements-xml ${ARG_FILES} | grep "<replacement " > /tmp/log && exit 1 || exit 0
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
    COMMENT "Checking ${_target} code formatting with clang-format"
    VERBATIM
  )
  add_dependencies(check-format check-format-${_target})
endfunction()

function(add_tidy_target _target)
  if(NOT CLANG_TIDY_PROG)
    return()
  endif()
  cmake_parse_arguments(ARG "" "" "FILES;DEPENDS" ${ARGN})

  add_custom_target(tidy-${_target}
    COMMAND ${PARALLEL_PROG} ${CLANG_TIDY_PROG} -fix -p=${CMAKE_BINARY_DIR} ${PARALLEL_DELIMITER} ${ARG_FILES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
    DEPENDS ${ARG_DEPENDS}
    COMMENT "Running clang-tidy for ${_target}"
    VERBATIM
  )
  add_dependencies(tidy tidy-${_target})

  add_custom_target(check-tidy-${_target}
  COMMAND ${PARALLEL_PROG} ${CLANG_TIDY_PROG} -p=${CMAKE_BINARY_DIR} ${PARALLEL_DELIMITER} ${ARG_FILES} | grep . && exit 1 || exit 0
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
    DEPENDS ${ARG_DEPENDS}
    COMMENT "Running clang-tidy for ${_target}"
    VERBATIM
  )
  add_dependencies(check-tidy check-tidy-${_target})
endfunction()
