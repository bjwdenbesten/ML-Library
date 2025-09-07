cmake_minimum_required(VERSION 3.14)
project(MLLibrary)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


file(GLOB SOURCES "src/*.cpp")
add_library(MLLibrary STATIC ${SOURCES})


target_include_directories(MLLibrary PUBLIC include)

set(CMAKE_PREFIX_PATH "/opt/homebrew")
find_package(Eigen3 CONFIG REQUIRED)
message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "Eigen3 version: ${EIGEN3_VERSION}")
target_link_libraries(MLLibrary PUBLIC Eigen3::Eigen)

file(GLOB EXAMPLES "training_examples/*.cpp")
file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})

foreach(EXAMPLE ${EXAMPLES})
    get_filename_component(NAME ${EXAMPLE} NAME_WE)
    add_executable(${NAME} ${EXAMPLE})
    target_link_libraries(${NAME} PRIVATE MLLibrary)
endforeach()