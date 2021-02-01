cmake_minimum_required(VERSION 3.10.2)
project(ScoringSystem)

add_executable(${PROJECT_NAME} main.cc)

find_package(Ceres REQUIRED)

target_include_directories(${PROJECT_NAME}
    PRIVATE ${CERES_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE ${CERES_LIBRARIES}
)