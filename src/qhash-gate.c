#include "qhash-gate.h"

bool register_qhash_algo(algo_gate_t *gate)
{
    gate->hash = &qhash_hash;
    gate->miner_thread_init = &qhash_thread_init;
    // gate->optimizations = ()
    return true;
}