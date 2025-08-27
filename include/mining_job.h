#ifndef MINING_JOB_H
#define MINING_JOB_H

#include <string>
#include <vector>

struct MiningJob {
    std::string job_id;
    std::string prev_hash;
    std::string coinb1;
    std::string coinb2;
    std::vector<std::string> merkle_branches;
    std::string version;
    std::string nbits;
    std::string ntime;
    bool clean_jobs;
};

#endif // MINING_JOB_H