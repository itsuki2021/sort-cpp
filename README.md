# sort-cpp

C++ implementation of SORT (Simple, online, and realtime tracking of multiple objects in a video sequence).

## Introduction

SORT was initially described in [this paper](http://arxiv.org/abs/1602.00763) with the [python code](https://github.com/abewley/sort), this repo try to reproduce it by using C++.

## pre-install

[opencv 3.4.16](https://docs.opencv.org/3.4/index.html)

## run
````shell
mkdir build && cd build
cmake ..
make
./demo_sort
````
