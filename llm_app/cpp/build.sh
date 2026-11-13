#!/bin/sh
set -eu
cd "$(dirname "$0")"
# Загрузчик проверяет X86_V4 и F16C до вызова библиотечных ядер.
case "$(uname -s)" in
  Linux) qmm_output=qmm.so; ${CXX:-g++} -O3 -march=x86-64-v4 -mf16c -std=c++17 -pthread -fPIC -shared -o qmm.so qmm.cpp ;;
  Darwin) qmm_output=qmm.dll; x86_64-w64-mingw32-g++ -O3 -march=x86-64-v4 -mf16c -std=c++17 -shared -static -o qmm.dll qmm.cpp ;;
  *) echo "Используйте Linux g++ или кросс-сборку Windows с Mac" >&2; exit 1 ;;
esac

python3 write_manifest.py "$qmm_output"
