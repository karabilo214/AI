#!/bin/sh
set -eu
exec sh "$(dirname "$0")/../../ramvi/engine/ramvi_engine/native/build.sh" "$@"
