# ------------------------------------------------------------------
# Windows:
#   cd build
#   cmake -G "NMake Makefiles" -S ..
#   nmake
# ------------------------------------------------------------------
# python -c "import numpy as np; import pymacosf90 as f; "
# ------------------------------------------------------------------

cmake_minimum_required(VERSION 3.31) # Needed to avoid requiring embedded Python libs too

project(pymacosf90
  VERSION 1.0
  DESCRIPTION "Python Interface"
  LANGUAGES Fortran C CXX
)
enable_language(Fortran C CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/..)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)
# add_subdirectory(${PROJECT_SOURCE_DIR}/source)


# Release or Debug: set to Release
# Set the Build type to Release
# if(NOT CMAKE_BUILD_TYPE)   # default is Debug
  set(CMAKE_BUILD_TYPE Release CACHE STRING " " FORCE)
# endif()
# Set the Build type to Release
# set(CMAKE_BUILD_TYPE Release)
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")


# set(CMAKE_C_COMPILER "cl")  # Visual C compiler
# set(CMAKE_F_COMPILER "ifx")  # Intel Fortran compiler



# Safety net
if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there.\n"
  )
endif()

# Grab Python, 3.13 or newer
find_package(Python 3.13 REQUIRED
  COMPONENTS Interpreter Development.Module NumPy)

# Grab the variables from a local Python installation
# F2PY headers
execute_process(
  COMMAND "${Python_EXECUTABLE}"
  -c "import numpy.f2py; print(numpy.f2py.get_include())"
  OUTPUT_VARIABLE F2PY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "\\" "/" F2PY_INCLUDE_DIR  "${F2PY_INCLUDE_DIR}")  # Fix path formatting for Win

# set(PYTHON_LIBRARIES Python3::Python)
# if (MSVC)
#     set(PYTHON_LIBRARIES ${PYTHON_LIBRARIES} ${PYTHON_LIBRARIES}/python312.lib)
#     # set(PYTHON_LIBRARIES ${PYTHON_LIBRARIES} python312.lib)
#     # set_target_properties(fibby PROPERTIES DEBUG_POSTFIX "_d")
# endif()
# Python3_EXECUTABLE_RELEASE


# -------------------------------------------
# Fortran: Compiler options for all
# -------------------------------------------
set(CMAKE_Fortran_COMPILER "ifx")
# get_filename_component(Fortran_COMPILER_NAME ${CMAKE_Fortran_COMPILER} NAME)

# update Fortran Flags: default: /nologo /fpp
# => initialised by "FFLAGS" Environment settings
# set(CMAKE_Fortran_FLAGS ${CMAKE_Fortran_FLAGS} " /free /names:uppercase /assume:underscore")
# set(CMAKE_Fortran_FLAGS "/nologo /fpp /free /names:uppercase /assume:nounderscore")
set(CMAKE_Fortran_FLAGS "/nologo /fpp /Qparallel /free /O2 /gen-interfaces /fp:strict /names:uppercase /assume:nounderscore")
# /Qm64 /Qmkl /Qopt-mem-layout-trans:0
# add_compile_definitions($<COMPILE_LANGUAGE:Fortran>: "/free")
message(STATUS "CMAKE_Fortran_FLAGS: ${CMAKE_Fortran_FLAGS}\n")
# set(CMAKE_F_COMPILER "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/bin")



# Print out the discovered paths -- for debugging
include(CMakePrintHelpers)
# cmake_print_variables(Python_INCLUDE_DIRS)
# cmake_print_variables(F2PY_INCLUDE_DIR)
# cmake_print_variables(Python_NumPy_INCLUDE_DIRS)
# cmake_print_variables(WIN32)
# cmake_print_variables(PYTHON_LIBRARIES)
# cmake_print_variables(Python_EXECUTABLE)
# cmake_print_variables(Python_INTERPRETER)
# cmake_print_variables(Python_Interpreter_FOUND)
# cmake_print_variables(Python_FOUND)
# cmake_print_variables(Python_EXTENSION_MODULE_SUFFIX)
# cmake_print_variables(CMAKE_F_COMPILER)
# cmake_print_variables(CMAKE_Fortran_FLAGS)
# cmake_print_variables(F2PY_FCOMPILER)
# cmake_print_variables(F2PY_SUFFIX)
# cmake_print_variables(_fcompiler_opts)
# cmake_print_variables(PROJECT_NAME)



# Common variables
set(f2py_module_name ${PROJECT_NAME})
set(pymacos_src_dir "${CMAKE_SOURCE_DIR}/source")

set(fortran_src_file     ${pymacos_src_dir}/pymacos.f90)
set(fortran_src_file_fpp ${CMAKE_CURRENT_BINARY_DIR}/pymacos.i90)
set(f2py_module_c        "${f2py_module_name}module.c")
set(f2py_wrapper_pyf     "${CMAKE_CURRENT_BINARY_DIR}/${f2py_module_name}.pyf")
set(f2py_wrapper_f90     "${CMAKE_CURRENT_BINARY_DIR}/${f2py_module_name}-f2pywrappers2.f90")

set(macos_mod_dir "${CMAKE_SOURCE_DIR}/../../../../../macos_open_source/cmake/build/mod")
set(macos_lib_dir "${CMAKE_SOURCE_DIR}/../../../../../macos_open_source/cmake/build/libs")

set_source_files_properties(
  ${fortran_src_file}
  PROPERTIES Fortran_PREPROCESS ON)


add_custom_command(
  OUTPUT   ${fortran_src_file_fpp}
  COMMAND "ifx" /P  ${fortran_src_file}
  DEPENDS ${fortran_src_file}
)
# -------------------------------------------------------


# Generate sources
add_custom_target(
  genpyf
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${f2py_module_c}"
  ${f2py_wrapper_pyf}
)


add_custom_command(
  OUTPUT ${f2py_wrapper_pyf}

  COMMAND ${Python_EXECUTABLE}  -m "numpy.f2py" -h ${f2py_wrapper_pyf}
            -m ${PROJECT_NAME} ${fortran_src_file_fpp}
            --no-lower --overwrite-signature --verbose  --skip-empty-wrappers

  DEPENDS ${fortran_src_file_fpp}

  VERBATIM
)



add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${f2py_module_c}"
         ${f2py_wrapper_f90}

  COMMAND ${Python_EXECUTABLE}  -m "numpy.f2py"
                -m ${f2py_module_name}
                   ${fortran_src_file_fpp}
                   --no-lower # Important for Win -- Linux may differ
                   --verbose
                   --skip-empty-wrappers
  DEPENDS ${fortran_src_file_fpp} ${f2py_wrapper_pyf}  # Fortran source

  VERBATIM
)

# Set up target
Python_add_library(${CMAKE_PROJECT_NAME} MODULE  WITH_SOABI  # SOABI (Shared Object Application Interface)
  ${f2py_module_c}                        # Generated
  ${f2py_wrapper_pyf}                     # Generated
  ${f2py_wrapper_f90}                     # Generated -- needed when modules are present in fortran file
  "${F2PY_INCLUDE_DIR}/fortranobject.c"   # From NumPy
  "${fortran_src_file}"                   # Fortran source(s)
)

target_compile_options(${CMAKE_PROJECT_NAME} PUBLIC -DNO_APPEND_FORTRAN -DUPPERCASE_FORTRAN)

target_include_directories(${CMAKE_PROJECT_NAME}
        PRIVATE
        ${F2PY_INCLUDE_DIR}
        ${Python_NumPy_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR}
        ${macos_lib_dir}
        ${macos_mod_dir}
        ${pymacos_src_dir}
      )

# Depend on sources
target_link_libraries(${CMAKE_PROJECT_NAME}
        PRIVATE
        Python::NumPy
        ${macos_lib_dir}/blaslib.lib
        ${macos_lib_dir}/lapacklib.lib
        ${macos_lib_dir}/npsol.lib
        ${macos_lib_dir}/smacos.lib
)

add_dependencies(${CMAKE_PROJECT_NAME} genpyf)
