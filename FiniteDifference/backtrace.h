#ifndef BACKTRACE_H
#define BACKTRACE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Demangle C++ symbols and return ptr to entire result, unbounded. (scary!) */
char* demangle2(char *bt);

/* Demangle C++ symbols in *bt and save n bytes of the result in *dest. */
size_t demangle(char *dest, char *bt, size_t n);

/* Write a backtrace of depth size into *buf, return buf */
char* get_backtrace(char *buf, size_t size);

#ifdef __cplusplus
} /* extern C */
#endif

#endif /* end of include guard */
