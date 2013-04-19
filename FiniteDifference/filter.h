#ifndef FILTER_H
#define FILTER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <unistd.h>
#include <sys/types.h>

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

typedef struct Piped_Child {
    pid_t pid;
    int in, out; // You write to in and read from out
} piped_child;

int piped_child_init(piped_child *p);

int piped_child_close(piped_child *p);

int fixed_read(char * const * const dest, size_t size, int src);

int dynamic_read(char **dest, int src);

ssize_t filter(char ** const dest, size_t const osize,
        void const * const src, size_t const isize,
        size_t const argc, char * const argv[]);

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* end of include guard */
