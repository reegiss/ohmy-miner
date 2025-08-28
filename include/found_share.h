#ifndef FOUND_SHARE_H__
#define FOUND_SHARE_H__

#include <string>

struct FoundShare {
    std::string job_id;
    std::string extranonce2;
    std::string ntime;
    std::string nonce_hex;
};

#endif // FOUND_SHARE_H__