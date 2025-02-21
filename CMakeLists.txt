cmake_minimum_required(VERSION 3.10)
project(faiss_demo)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)
find_package(faiss REQUIRED)

add_executable(faiss_demo vector_searcher/tests/main.cpp)
target_link_libraries(faiss_demo PRIVATE faiss)
