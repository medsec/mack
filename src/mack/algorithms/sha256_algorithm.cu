#include <mack/algorithms/sha256_algorithm.cuh>
#include <cuda_runtime.h>

namespace mack{

__device__
sha256_Algorithm::sha256_Algorithm()
: _target_size(32)
{
}

__constant__ const uint32_t H[]={0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

__device__ __host__
void
sha256_Algorithm::compute_target(mack::core::candidate key_candidate, unsigned char* result) const
{

	  const uint32_t k[]={
	   0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	   0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	   0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	   0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	   0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	   0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	   0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	   0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
	  uint32_t w[64]={0};

	  uint32_t cand[16];
	  memset(cand,0,16*sizeof(uint32_t));

	  //conversion to uint32 and padding
	  uint32_t dl=key_candidate.length,j;
	  for(j=0;j<dl;j++){
	      uint32_t tmp=0;
	      tmp |= (((uint32_t) key_candidate.value[j]) << ((3-(j & 0x3)) << 3));
	      cand[j/4]|=tmp;
	    }
	  cand[j / 4] |= (((uint32_t) 0x80) << ((3-(j & 0x3)) << 3));
	  cand[15]=0x00000000|(dl*8);


	  #pragma unroll 64
	  for(uint32_t j=0;j<64;j++){
	    if(j<16) w[j]=cand[j];
	  else w[j]=sigma1(w[j-2])+w[j-7]+sigma0(w[j-15])+w[j-16];
	  }

	  uint32_t a=H[0];uint32_t b=H[1];uint32_t c=H[2];uint32_t d=H[3];
	  uint32_t e=H[4];uint32_t f=H[5];uint32_t g=H[6];uint32_t h=H[7];
	  #pragma unroll 64
	  for(uint32_t j=0;j<64;j++){
	   uint32_t t1=h+Sigma1(e)+Ch(e,f,g)+k[j]+w[j];
	   uint32_t t2=Sigma0(a)+Maj(a,b,c);
	   h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
	  }

	result[0] = a+H[0] >> 24;
	result[1] = a+H[0] >> 16;
	result[2] = a+H[0] >>  8;
	result[3] = a+H[0];
	result[4] = b+H[1] >> 24;
	result[5] = b+H[1] >> 16;
	result[6] = b+H[1] >>  8;
	result[7] = b+H[1];
	result[8] = c+H[2] >> 24;
	result[9] = c+H[2] >> 16;
	result[10] = c+H[2] >> 8;
	result[11] = c+H[2];
	result[12] = d+H[3] >> 24;
	result[13] = d+H[3] >> 16;
	result[14] = d+H[3] >>  8;
	result[15] = d+H[3];
	result[16] = e+H[4] >> 24;
	result[17] = e+H[4] >> 16;
	result[18] = e+H[4] >>  8;
	result[19] = e+H[4];
	result[20] = f+H[5] >> 24;
	result[21] = f+H[5] >> 16;
	result[22] = f+H[5] >>  8;
	result[23] = f+H[5];
	result[24] = g+H[6] >> 24;
	result[25] = g+H[6] >> 16;
	result[26] = g+H[6] >>  8;
	result[27] = g+H[6];
	result[28] = h+H[7] >> 24;
	result[29] = h+H[7] >> 16;
	result[30] = h+H[7] >>  8;
	result[31] = h+H[7];

}

__device__ __host__
unsigned int
sha256_Algorithm::get_target_size(size_t length) const
{
  return _target_size;
}

__device__ __host__
unsigned char*
sha256_Algorithm::get_name() const
{
	return (unsigned char*)"SHA256";
}


__device__ __host__
sha256_Algorithm::~sha256_Algorithm() { }

}

