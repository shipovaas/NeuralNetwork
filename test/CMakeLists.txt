include_directories("../eigen")
include_directories("../EigenRand")
include_directories("../src")
include_directories("../googletest")

file(GLOB NET_SOURCES "../src/*.cpp")
file(GLOB TEST_SOURCES "*.cpp")

add_executable(Tests ${TEST_SOURCES} ${NET_SOURCES} mnist_train.h except.h except.cpp)
target_link_libraries(Tests gtest)