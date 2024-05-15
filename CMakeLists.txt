cmake_minimum_required(VERSION 3.0)
project(Neural-Networks)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#Set(CMAKE_BUILD_TYPE Release)

add_subdirectory(eigen)
add_subdirectory(EigenRand)
add_subdirectory(googletest)
add_subdirectory(src)
add_subdirectory(test)






#cmake_minimum_required(VERSION 3.25)
#project(NeuralNetwork)
#
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_VERBOSE_MAKEFILE ON)
#
#
#include_directories(EigenRand)
#
#add_executable(neural_network src/main.cpp src/ActivationFunction.cpp src/Layer.cpp src/LossFunction.cpp src/NeuralNetwork.cpp)
#add_subdirectory(googletest)
#add_subdirectory(eigen)
#add_subdirectory(EigenRand)
#
#add_executable(run_tests src/maintest.cpp)
#target_link_libraries(run_tests gtest eigen eigenrand)
#include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR} ${gtest_BINARY_DIR})


#add_subdirectory(googletest)
#add_subdirectory(eigen)target_link_libraries(run_tests gtest gtest_main)
#add_subdirectory(EigenRand)
#cmake_minimum_required(VERSION 3.14)
#
#project(BasicNeuralNetwork)
#
#set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#
#add_subdirectory(libs/googletest)
#
#include(src/sources.cmake)
#include(tests/sources.cmake)
#
#enable_testing()
#
#target_link_libraries(
#        test_neural_network
#        neural_network
#        GTest::gtest_main
#)
#
#include(GoogleTest)
#gtest_discover_tests(test_neural_network)


#cmake_minimum_required(VERSION 3.15)