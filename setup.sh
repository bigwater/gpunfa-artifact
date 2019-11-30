#!/bin/sh
# Sets up environment for artifact evaluation

echo ${PWD}
export GPUNFA_ROOT=${PWD}

if [ ! -d "gpunfa_benchmarks" ]; then
    echo "Downloading benchmarks (795MB)... "
    wget --quiet --show-progress -O gpunfa_benchmarks.zip https://www.dropbox.com/s/qn9skx7374q1x6u/gpunfa_benchmarks.zip

    if [ $? -eq 0 ]; then
        echo "unzip the benchmarks... "
        unzip -q gpunfa_benchmarks.zip
    else
        echo "download apps failed. "
        exit 1
    fi
else
   echo "Skip downloading benchmarks since they exist. "	
fi

if [ ! -d "raw_data" ]; then
    echo "Downloading the raw data for generating all experimental results shown in the paper... (430MB)"
    wget --quiet --show-progress -O raw_data.zip https://www.dropbox.com/s/8bym4qa9xeepkkf/raw_data.zip

    if [ $? -eq 0 ]; then
        echo "unzip the raw data... "
        unzip -q raw_data.zip
    else
        echo "download raw data failed. "
        exit 1
    fi
else
    echo "skip downloading raw data since it exists. "
fi


echo "Build from the source code. "
cd gpunfa_code && rm -rf build && mkdir build
cd build && cmake --quiet -DCMAKE_BUILD_TYPE=Release ..


if [ $? -eq 0 ]; then
	make -j
else
        echo "cmake error"
        exit 1
fi

cd ${GPUNFA_ROOT}

echo "finished build the executables. Set them to PATH. "
export PATH="${GPUNFA_ROOT}/gpunfa_code/build/bin:${PATH}"

echo "Replace the paths in config files. "
sed -i "s|/home/hyliu/gpunfa_benchmarks|${GPUNFA_ROOT}/gpunfa_benchmarks|g" ${GPUNFA_ROOT}/gpunfa_code/scripts/configs/app_spec









