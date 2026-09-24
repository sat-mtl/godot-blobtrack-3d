#! /usr/bin/bash

# This works on arch linux. it does not work on gcc

if [[ -z "${BUILD_RELEASE}" ]]; then
    # if BUILD_RELEASE has no value, build a debug version
    cmake -B build
else
    # else, build a release version
    cmake -B build -DGODOTCPP_TARGET="template_release" -DCMAKE_BUILD_TYPE=Release
fi
make -sC build -j8

cp build/blobtrack3d.linux.template* ./bin/linux/
