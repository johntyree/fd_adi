#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <sstream>
#include <stdexcept>

#include "backtrace.h"


#define TRACE debug_printer(std::cout, "TRACE", __FILE__, __PRETTY_FUNCTION__, __LINE__ , std::string());

#define LOG(msg) {std::ostringstream s; s << msg;\
    debug_printer(std::cout, "LOG", __FILE__, __PRETTY_FUNCTION__, __LINE__ , s.str());\
}

#define DIE(msg) {\
    std::ostringstream s;\
    unsigned const sz = 512 * 1024;\
    char full[sz];\
    get_backtrace(full, sz);\
    s << __FILE__ << "(" << __LINE__ << ")\n"\
    << __PRETTY_FUNCTION__ << "\n\t"\
    << msg << "\n"\
    << full << "\n";\
    throw std::domain_error(s.str());\
}

#define ENDL std::cout << std::endl

#define FULLTRACE noop();
static inline void noop() {};

std::string const FOLDED = "FOLDED";
std::string const CAN_FOLD = "CAN_FOLD";
std::string const CANNOT_FOLD = "CANNOT_FOLD";

static inline void debug_printer(std::ostream &os, const char *type, const char *fn, const char *func, int line, std::string msg) {
    os
        << type << ": "
        << fn << "(" << line << "): "
        << func;
        if (msg.size() != 0) {
            os << "\n\t" << msg;
        }
        os << std::endl;
}


typedef double REAL_t;
typedef long int Py_ssize_t;


template <typename T>
void cout(T const &a) {
    std::cout << a;
}


template <typename T>
std::string to_string(T const &a) {
    std::ostringstream s;
    s << a;
    return s.str();
}

#endif /* end of include guard */
