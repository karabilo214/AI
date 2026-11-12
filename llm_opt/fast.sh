#!/bin/bash
# Запуск с быстрыми потоками Numba: OpenMP из Homebrew, потоки ждут работу активно 200 мс (запуск 3 мкс вместо 42).
# Пример: ./fast.sh bench_numba.py
cd "$(dirname "$0")"
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib NUMBA_THREADING_LAYER=omp KMP_BLOCKTIME=200 NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-8}
exec ../.venv/bin/python -u "$@"
