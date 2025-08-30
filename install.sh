#!/bin/bash

# Linux build
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && make && ./qtcminer --algo qhash --url qubitcoin.luckypool.io:8610 --user bc1qacadts4usj2tjljwdemfu44a2tq47hch33fc6f.R3G --pass x