#ifndef SANDIA_TABLE_CUH
#define SANDIA_TABLE_CUH

// Function that drive the tables reading between CPU and GPU
__host__ __device__ unsigned short int PhotoElec_std_NbIntervals(unsigned int pos);
__host__ __device__ unsigned short int PhotoElec_std_CumulIntervals(unsigned int pos);
__host__ __device__ float PhotoElec_std_ZtoAratio(unsigned int pos);
__host__ __device__ float PhotoElec_std_IonizationPotentials(unsigned int pos);
__host__ __device__ float PhotoElec_std_SandiaTable(unsigned int pos, unsigned int id);

#endif
