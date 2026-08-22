cmake_minimum_required(VERSION 3.10)
project(t_hybrid_cli)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(THYBRID_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/T-Hybrid-planner/src/path_planner")

find_package(PkgConfig REQUIRED)
pkg_check_modules(OMPL REQUIRED ompl)

add_executable(t_hybrid_plan
  t_hybrid_plan.cpp
  ${THYBRID_ROOT}/src/algorithm.cpp
  ${THYBRID_ROOT}/src/node2d.cpp
  ${THYBRID_ROOT}/src/node3d.cpp
  ${THYBRID_ROOT}/src/collisiondetection.cpp
  ${THYBRID_ROOT}/src/dubins.cpp
)

target_include_directories(t_hybrid_plan PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${THYBRID_ROOT}/include
  /usr/include/eigen3
  /opt/ros/noetic/include
  ${OMPL_INCLUDE_DIRS}
)

target_link_directories(t_hybrid_plan PRIVATE ${OMPL_LIBRARY_DIRS})
target_link_libraries(t_hybrid_plan PRIVATE ${OMPL_LIBRARIES})
target_compile_options(t_hybrid_plan PRIVATE ${OMPL_CFLAGS_OTHER})
