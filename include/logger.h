#ifndef LOGGER_H_
#define LOGGER_H_

#include <iostream>
#include <sstream>

class Logger {
public:
    template<typename... Args>
    static void info(Args&&... args) {
        std::cout << "[INFO] " << concat(std::forward<Args>(args)...) << std::endl;
    }

    template<typename... Args>
    static void warn(Args&&... args) {
        std::cerr << "[WARN] " << concat(std::forward<Args>(args)...) << std::endl;
    }

    template<typename... Args>
    static void error(Args&&... args) {
        std::cerr << "[ERROR] " << concat(std::forward<Args>(args)...) << std::endl;
    }

private:
    template<typename... Args>
    static std::string concat(Args&&... args) {
        std::ostringstream oss;
        (oss << ... << args);  // fold expression (C++17+)
        return oss.str();
    }
};

#endif // LOGGER_H_
