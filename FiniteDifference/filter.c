#ifdef __cplusplus
extern "C" {
#endif

#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <assert.h>

#include "filter.h"

int piped_child_init(piped_child *p) {
    /* Initialize a piped_child by fork()ing and storing the pid. Then open two
     * unidirectional pipes and store the input of one and output of the other
     * such that the parent's input goes to the child's output and vice-versa.
     *
     * It is the caller's responsibility to call piped_child_close() to avoid
     * dangling file-descriptors.
     */
    if (!p) {return -1;}

    int par[2], child[2];
    if (pipe(par) == -1 || pipe(child) == -1) {return -1;}
    if ((p->pid = fork()) == -1) {return -1;}

    if (!p->pid) { /* child */
        p->in = dup2(par[1], STDOUT_FILENO);
        p->out = dup2(child[0], STDIN_FILENO);
        close(child[0]); close(par[0]);
        close(child[1]); close(par[1]);
    } else { /* parent */
        p->in = child[1];
        p->out = par[0];
        close(child[0]); close(par[1]);
    }

    return 0;
}

int piped_child_close(piped_child *p) {
    close(p->in);
    close(p->out);
    return 0;
}

int fixed_read(char * const * const dest, size_t size, int src) {
    ssize_t bytes;
    size_t total = 0;
    while ((bytes = read(src, *dest + total, (size - total))) > 0) {
        total += bytes;
    }
    if (bytes == -1) {return -1;}
    if (total < size) {
        (*dest)[total] = 0;
    }
    return total;
}

int dynamic_read(char ** const dest, int src) {
    ssize_t bytes;
    size_t total = 0;
    size_t const bufsize = 8;
    size_t size = bufsize;
    void *buf = malloc(bufsize);
    while ((bytes = read(src, buf, bufsize)) > 0) {
        // Allocate extra byte for terminiating NULL
        if (total+bytes+1 > size) {
            size = (total+bytes) * 1.5;
            *dest = (char *)realloc(*dest, size);
            if (!*dest) {
                errno = 0;
                return -1;
            }
        }
        memcpy(*dest + total, buf, bytes);
        total += bytes;
    }
    if (bytes == -1) {return -1;}
    (*dest)[total] = 0;
    free(buf);
    return total;
}

ssize_t filter(char ** const dest, size_t const osize,
        void const * const src, size_t const isize,
        size_t const argc, char * const argv[]) {
    /*
     * filter - Pipe the buffer src through a process specified by argv[].
     *
     * src is a buffer from which isize bytes will be read and piped into
     * stdin of argv[0].
     *
     * *dest is a buffer where at most osize bytes of output from stdout of
     * argv[0] will be stored. If osize > output then dest will terminate
     * with a NULL byte as in dest[output_size] = NULL. This is not the case
     * if output is truncated.
     *
     * If osize is <= 0, *dest will be resized to fit tne entire output. In
     * that case *dest must be valid for realloc().
     *
     * argv[] is as specified in execv(3), with the exception that the path to
     * the executable is included as argv[0], shifting other arguments back by
     * one. This MUST terminate with a NULL ptr to indicate the end of the
     * argument list.
     *
     * The number of bytes written to *dest is returned on success. A return
     * value of -1 and possible non-zero errno indicates error.
     *
     * This example is analogous to the unix command
     *      cat -A -v < "Some input." | head -c 1024 > res
     *
     * Example:
     *      char *res = malloc(1024);
     *      char *in = "\n\nSome input.\n";
     *
     *                      // path,    pname, args...,    NULL
     *      char *argv[] = {"/bin/cat", "cat", "-A", "-v", NULL};
     *      int argc = 5;
     *
     *      int size = filter(&res, 1024, in, strlen(in), argc, argv);
     *      if (size == -1) {
     *          perror("Pipe failed");
     *          exit(EXIT_FAILURE);
     *      }
     *      printf("Wrote %i bytes.\n", size);
     *      fwrite(res, size, 1, stdout);
     *
     *
     *
     * This shows the auto growing dest buffer.
     *      cat -A -v < "Some input." > res
     *
     * Example:
     *      char *res = NULL;
     *      ... // same as above ..
     *      int size = filter(&res, 0, in strlen(in), argc, argv);
     *      ....
     */
    short can_grow = osize <= 0;
    short fail = 1;
    size_t i;
    for (i = 0; i < argc; ++i) {
        if (argv[i] == NULL) {
            fail = 0;
            break;
        }
    }
    if (fail) {
        errno = EINVAL;
        return -1;
    }

    piped_child p;
    if (piped_child_init(&p) != 0) {return -1;}

    /* child process */
    if (!p.pid) {
        /* fprintf(stderr, "Child: Launching process.\n"); */
        /* fflush(stderr); */
        int ret = execv(argv[0], argv+1);
        /* fprintf(stderr, "Child: Process returned.\n"); */
        /* fflush(stderr); */
        piped_child_close(&p);
        exit(ret);
    }
    /* end child process */

    /* parent process */
    /* fprintf(stderr, "Parent: Writing...\n"); */
    if (write(p.in, src, isize) != (ssize_t)isize) return -1;
    close(p.in);

    ssize_t total = 0;
    /* fprintf(stderr, "Parent: Reading...\n"); */
    if (can_grow) {
        if ((total = dynamic_read(dest, p.out)) == -1) {return -1;}
    } else {
        if ((total = fixed_read(dest, osize, p.out)) == -1) {return -1;}
    }

    int status;
    waitpid(p.pid, &status, 0);
    if (WIFEXITED(status)) {
        if (WEXITSTATUS(status) != 0) {
            errno = 0;
            return -1;
        }
    }
    piped_child_close(&p);
    return total;
}


#ifdef __cplusplus
} // extern "C"
#endif
