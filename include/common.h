/************************************************************
 * @file common.h
 * @brief Common macros and definitions for the project.
 *
 * This header defines utility macros for compiler optimizations.
 ************************************************************/

#ifndef COMMON_H__
#define COMMON_H__

/**
 * @def unlikely(x)
 * @brief Macro to provide branch prediction information to the compiler.
 *
 * On GCC and Clang, this macro uses __builtin_expect to indicate that
 * the expression 'x' is unlikely to be true, which can help the compiler
 * optimize branch prediction. On other compilers, it evaluates to 'x'.
 *
 * @param x Expression to be evaluated for likelihood.
 */
#if defined(__GNUC__) || defined(__clang__)
#  define unlikely(x) __builtin_expect(!!(x), 0)
#else
#  define unlikely(x) (x)
#endif

#endif // COMMON_H__