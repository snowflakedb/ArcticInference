#include "kernel_utils.h"

#define TOP_K_SWITCH(N_TOP_K, ...)         \
    [&] {                                  \
        if (1 == N_TOP_K) {                \
            constexpr int CONST_TOP_K = 1; \
            __VA_ARGS__();                 \
        } else if (2 == N_TOP_K) {         \
            constexpr int CONST_TOP_K = 2; \
            __VA_ARGS__();                 \
        } else if (4 == N_TOP_K) {         \
            constexpr int CONST_TOP_K = 4; \
            __VA_ARGS__();                 \
        } else if (8 == N_TOP_K) {         \
            constexpr int CONST_TOP_K = 8; \
            __VA_ARGS__();                 \
        }                                  \
    }()
