#ifndef MD5_HELPER_CUH_
#define MD5_HELPER_CUH_

#include <cuda_runtime.h>

namespace mack{

#define PLAIN_LENGTH 50

typedef unsigned int MD5_u32plus;

typedef struct {
	MD5_u32plus lo, hi;
	MD5_u32plus a, b, c, d;
	unsigned char buffer[64];
	MD5_u32plus block[16];
} MD5_CTX;

__host__ __device__ int strlenDevice(char* str);
__host__ __device__ void MD5_Init(MD5_CTX *ctx);
__host__ __device__ void* body(MD5_CTX *ctx, void *data, unsigned long size);
__host__ __device__ void MD5_Update(MD5_CTX *ctx, void *data, unsigned long size);
__host__ __device__ void MD5_PreFinal(MD5_CTX *ctx);
__host__ __device__ void MD5_Final(unsigned char *result, MD5_CTX *ctx);

}

#endif /* MD5_HELPER_CUH_ */
