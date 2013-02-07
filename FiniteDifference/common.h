#ifndef COMMON_H
#define COMMON_H


#define TRACE debug_printer("TRACE", __FILE__, __PRETTY_FUNCTION__, __LINE__ , std::string());
#define LOG(msg) {std::ostringstream s; s << msg; debug_printer("LOG", __FILE__, __PRETTY_FUNCTION__, __LINE__ , s.str());}
void debug_printer(const char *type, const char *fn, const char *func, int line, std::string msg);

#define ENDL std::cout << std::endl

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
