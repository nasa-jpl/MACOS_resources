# ------------------------------------------------------------------
#   => set up Intel oneAPI Env
#   => activate Python >3.13 with Numpy
#
# Windows:
#   cd build
#   cmake -G "NMake Makefiles" -S ..
#   nmake
#
# Linux:
#   cd build
#   rm -r *; cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCMAKE_Fortran_COMPILER=ifx -S ..
#   make
#   
# ------------------------------------------------------------------
# Test:  python -c "import pymacosf90"
# ------------------------------------------------------------------

cmake_minimum_required(VERSION 3.31)
include(CMakePrintHelpers)

project(pymacosf90
  VERSION 1.00.00
  DESCRIPTION "Python Interface"
  LANGUAGES Fortran C CXX
)
enable_language(Fortran C CXX)

# ---------------------------------------------
# Safety net
if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there.\n"
  )
endif()

if(NOT EXISTS "${CMAKE_SOURCE_DIR}/../macos")
  message(FATAL_ERROR
    "MACOS submodule not found. Run: git submodule update --init --recursive")
endif()

# --------------------------------------------- Module dir
set(mod_dir "${CMAKE_BINARY_DIR}/mod")
make_directory(${mod_dir})
set(CMAKE_Fortran_MODULE_DIRECTORY ${mod_dir})

# --------------------------------------------- OS Definition
if(WIN32)
  message("-------------------------------\nWindows x64 Settings\n-------------------------------")
  set(WINx64 TRUE)
  set(LINUX64 FALSE)

else() # if(CMAKE_SYSTEM_NAME STREQUAL "Linux")  # (CMAKE_HOST_UNIX)
  message("-------------------------------\nLinux Settings\n-------------------------------")
  set(WINx64 FALSE)
  set(LINUX64 TRUE)
endif()

# ------------------------------------------------------- Compiler
if (LINUX64)
  set(CMAKE_C_COMPILER icx)          # Intel oneAPI
  set(CMAKE_CXX_COMPILER icpx)       # Intel oneAPI
  set(CMAKE_Fortran_COMPILER ifx)    # Intel oneAPI
else()
  set(CMAKE_Fortran_COMPILER ifx)    # Intel oneAPI
  set(CMAKE_C_COMPILER cl)           # MS VC
  set(CMAKE_CXX_COMPILER cl)         # MS VC
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)


# ------------------------------------------------------- Export
# note: it will fail to create when pymacos DLL is still in use
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../pymacos)  # dump pyd file to pymacos


# ------------------------------------------------------- Build Type
# Set the Build type to Release  (Default is Debug)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Release CACHE STRING " " FORCE)     # cache may require upd.
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")


# ------------------------------------------------------- Python > 3.13  (requires numpy > 2.2)
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


# ------------------------------------------------------- Fortran Pre-Processing
set_source_files_properties(PROPERTIES Fortran_PREPROCESS ON)


# ------------------------------------------------------- Common definitions
set(f2py_module_name      ${PROJECT_NAME})
set(pymacos_src_dir      "${CMAKE_SOURCE_DIR}/source")

set(fortran_src_file      ${pymacos_src_dir}/pymacos.f90)
set(fortran_src_file_fpp  ${CMAKE_CURRENT_BINARY_DIR}/pymacos.i90)
set(f2py_module_c        "${f2py_module_name}module.c")
set(f2py_wrapper_pyf     "${CMAKE_CURRENT_BINARY_DIR}/${f2py_module_name}.pyf")
set(f2py_wrapper_f90     "${CMAKE_CURRENT_BINARY_DIR}/${f2py_module_name}-f2pywrappers2.f90")

cmake_path(GET CMAKE_SOURCE_DIR PARENT_PATH cmake_source_parent_dir)
set(macos_mod_dir "${cmake_source_parent_dir}/macos/cmake/build/mod")
set(macos_lib_dir "${cmake_source_parent_dir}/macos/cmake/build/libs")

# set(macos_mod_dir  "${CMAKE_SOURCE_DIR}/../macos/cmake/build/mod")
# set(macos_lib_dir  "${CMAKE_SOURCE_DIR}/../macos/cmake/build/libs")


# ------------------------------------------------------- SMACOS dependent libs
if(WINx64)
  set(SMACOS_Libs
        ${macos_lib_dir}/blaslib.lib
        ${macos_lib_dir}/lapacklib.lib
        ${macos_lib_dir}/npsol.lib
        ${macos_lib_dir}/smacos.lib)
else()
  set(SMACOS_Libs
        ${macos_lib_dir}/libsmacos.a
        ${macos_lib_dir}/libnpsol.a
        ${macos_lib_dir}/libblaslib.a
        ${macos_lib_dir}/liblapacklib.a)
endif()

# ------------------------------------------------------- Fortran Flags
if(WINx64)
  set(FFlags            "/nologo /fpp /Qparallel /free /O2 /gen-interfaces")
  set(FFlags  "${FFlags} /fp:strict /Qfp-speculation:strict")
  set(FFlags  "${FFlags} /Qm64 /Qmkl /Qparallel /Qopt-mem-layout-trans /QxHost")
  set(FFlags  "${FFlags} /traceback")
  set(FFlags  "${FFlags} /names:uppercase /assume:nounderscore")
else()
  set(FFlags            "-nologo -free -align -O2 -fpp -fPIC -gen-interfaces ")
  set(FFlags  "${FFlags} -fp-model=strict -fp-speculation=strict")
  set(FFlags  "${FFlags} -xHost -m64 -qmkl -qopt-matmul -qopt-mem-layout-trans")
  # set(FFlags  "${FFlags} -traceback -check bounds -check stack")
  set(FFlags  "${FFlags} -names lowercase -assume underscore")
endif()
# set(Fortran_FLAGS_Lib "${FFlags}")
set(CMAKE_Fortran_FLAGS "${FFlags}")


set_source_files_properties(
  ${fortran_src_file}
  PROPERTIES Fortran_PREPROCESS ON)

# ------------------------------------------------------- Fortran Pre-processing
add_custom_command(
  OUTPUT   ${fortran_src_file_fpp}
  COMMAND ${CMAKE_Fortran_COMPILER} -P  ${fortran_src_file}
  DEPENDS ${fortran_src_file}
)
# -------------------------------------------------------
if(WINx64)
  set(f2py_flag "--no-lower")
else()
  set(f2py_flag "--lower")
endif()


# Generate sources
add_custom_target(genpyf
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${f2py_module_c}"
  ${f2py_wrapper_pyf}
)


add_custom_command(
  OUTPUT ${f2py_wrapper_pyf}

  COMMAND ${Python_EXECUTABLE}  -m "numpy.f2py" -h ${f2py_wrapper_pyf}
            -m ${PROJECT_NAME}
               ${fortran_src_file_fpp}
               ${f2py_flag}  # --no-/lower
               --overwrite-signature
               --verbose
               --skip-empty-wrappers

  DEPENDS ${fortran_src_file_fpp}
  VERBATIM
)

add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${f2py_module_c}"
         ${f2py_wrapper_f90}

  COMMAND ${Python_EXECUTABLE}  -m "numpy.f2py"
                -m ${f2py_module_name}
                   ${fortran_src_file_fpp}
                   ${f2py_flag}  # --no-/lower
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

if(WINx64)
  target_compile_options(${CMAKE_PROJECT_NAME} PUBLIC -DNO_APPEND_FORTRAN -DUPPERCASE_FORTRAN)
endif()

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
        ${SMACOS_Libs}
)

add_dependencies(${CMAKE_PROJECT_NAME} genpyf)
