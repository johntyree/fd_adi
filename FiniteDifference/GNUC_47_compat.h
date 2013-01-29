#if defined(__GNUC__)

#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 6)

#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#endif

#endif
