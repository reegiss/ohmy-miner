#ifndef FOUND_SHARE_H__
#define FOUND_SHARE_H__

#include <string>

/**
 * @struct FoundShare
 * @brief Represents a found share in a mining job.
 *
 * This structure holds the necessary information for a found share,
 * including the job identifier, extranonce2, ntime, and nonce in hexadecimal format.
 */
struct FoundShare {
    /**
     * @brief The identifier of the mining job.
     */
    std::string job_id;

    /**
     * @brief The extranonce2 value associated with the share.
     */
    std::string extranonce2;

    /**
     * @brief The ntime value for the share.
     */
    std::string ntime;

    /**
     * @brief The nonce value in hexadecimal format.
     */
    std::string nonce_hex;
};

#endif // FOUND_SHARE_H__