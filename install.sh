#!/bin/bash

if [ ! -d build ]; then
    mkdir build
fi

# Linux build (Release) and run main target only
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && make -j ohmy-miner && ./ohmy-miner --algo qhash --url qubitcoin.luckypool.io:8610 --user bc1qacadts4usj2tjljwdemfu44a2tq47hch33fc6f.R3G --pass x --batch 64