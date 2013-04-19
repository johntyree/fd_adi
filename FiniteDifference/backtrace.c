#ifdef __cplusplus
extern "C" {
#endif


#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>

#include "filter.h"

#ifndef min
#define min(a,b) ((a < b) ? (a) : (b))
#endif

char* demangle2(char *bt) {
    char *output = NULL;
    char * const argv[] = {"/usr/bin/c++filt", "c++filt", NULL};
    int argc = 3;
    filter(&output, 0, bt, strlen(bt), argc, argv);
    return output;
}


size_t demangle(char *dest, char *bt, size_t n) {
    assert(dest);
    assert(bt);
    char *output = demangle2(bt);
    size_t s = strlen(output)+1;
    n = s > n ? n : s;
    memcpy(dest, output, n);
    free(output);
    return n;
}

char* get_backtrace(char *full, size_t size) {
    int j, nptrs;
    void *buffer[100];
    char **strings;

    nptrs = backtrace(buffer, size);
    // printf("backtrace() returned %d addresses\n", nptrs);

    /* The call backtrace_symbols_fd(buffer, nptrs, STDOUT_FILENO)
        would produce similar output to the following: */

    strings = backtrace_symbols(buffer, nptrs);
    if (strings == NULL) {
        perror("backtrace_symbols() failed.");
        exit(EXIT_FAILURE);
    }

    full[0] = '\0';
    for (j = 1; j < min(10, nptrs); j++) {
        strcat(full, strings[j]);
        strcat(full, "\n");
    }

    free(strings);
    char buf[size];
    demangle(buf, full, size);
    memcpy(full, buf, size);
    return full;
}

#ifdef __cplusplus
} /* extern C */
#endif
