#include <mack/algorithms/md5_algorithm.cuh>
#include <cuda_runtime.h>
#include <mack/algorithms/md5_helper.cuh>

namespace mack{

__device__
MD5_Algorithm::MD5_Algorithm()
: _target_size(16)
{
}

__device__ __host__
void
MD5_Algorithm::compute_target(mack::core::candidate key_candidate, unsigned char* result) const
{
    MD5_CTX             context;
    //unsigned char digest [16];

    MD5_Init( &context );
	MD5_Update( &context, (void*)(key_candidate.value), key_candidate.length);
    MD5_Final( result, &context );

}

__device__ __host__
unsigned int
MD5_Algorithm::get_target_size(size_t length) const
{
  return _target_size;
}

__device__ __host__
unsigned char*
MD5_Algorithm::get_name() const
{
	return (unsigned char*)"MD5";
}

__device__ __host__
void
MD5_Algorithm::init(boost::program_options::variables_map const& options) { }

__device__ __host__
MD5_Algorithm::~MD5_Algorithm() { }

}

