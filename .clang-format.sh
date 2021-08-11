#!/bin/bash

for f in $(ls src/*.{cpp,cu,h,cuh})
do
  if [ -f "$f" ] && [ "$f" != "src/RcppExports.cpp" ]
  then
    clang-format-9 -i "${f}"
  fi
done
