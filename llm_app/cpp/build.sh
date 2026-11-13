#!/bin/sh
# Сборка ядер процессора для Windows (кросс-компилятором с Mac/Linux): brew install mingw-w64
cd "$(dirname "$0")" && x86_64-w64-mingw32-g++ -O3 -march=znver4 -std=c++17 -shared -static -o qmm.dll qmm.cpp
