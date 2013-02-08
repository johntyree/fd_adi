#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <sstream>
#include <stdexcept>

#define TRACE debug_printer("TRACE", __FILE__, __PRETTY_FUNCTION__, __LINE__ , std::string());
#define LOG(msg) {std::ostringstream s; s << msg; debug_printer("LOG", __FILE__, __PRETTY_FUNCTION__, __LINE__ , s.str());}
#define DIE(msg) {std::ostringstream s; s << msg; throw std::domain_error(s.str());}
#define ENDL std::cout << std::endl

#define FULLTRACE noop();

inline void noop() {};

inline void debug_printer(const char *type, const char *fn, const char *func, int line, std::string msg) {
    std::cout
        << type << ": "
        << fn << "(" << line << "): "
        << func;
        if (msg.size() != 0) {
            std::cout << "\n\t" << msg;
        }
        std::cout << std::endl;
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
