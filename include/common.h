#ifndef COMMON_H__
#define COMMON_H__

#if defined(__GNUC__) || defined(__clang__)
#  define unlikely(x) __builtin_expect(!!(x), 0)
#else
#  define unlikely(x) (x)
#endif

#endif // COMMON_H__