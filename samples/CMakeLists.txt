cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(samples)

cuda_add_executable(simple_example EXCLUDE_FROM_ALL simple_example/main.cu)
# cuda_add_executable(sowerby sowerby/main.cu)
# cuda_add_executable(xor_example xor_example/main.cu)
