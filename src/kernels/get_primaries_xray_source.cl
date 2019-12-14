#include "GGEMS/opencl/use_double_precision.hh"

//typedef __attribute__((aligned (1))) struct Units_t
//{
//__constant double a = 4.0;
//}Units;

//Units unit = {4.0};
/*#ifdef __cpluscplus
namespace Toto {
#endif
  static double const a = 4.0;
#ifdef __cpluscplus
}
#endif*/

__kernel void get_primaries_xray_source(
  __global double const* p_cdf,
  __global double const* p_energy_spectrum,
  unsigned int const number_of_energy_bins)
{
  // Get the index of thread
  int const kGlobalIndex = get_global_id(0);

  if (kGlobalIndex == 2) {
    printf("Particle id: %d\n", kGlobalIndex);
    printf("Number of energy bin: %u\n", number_of_energy_bins);
    for (unsigned int i = 0; i < number_of_energy_bins; ++i) {
      printf("Energy bin: %u, energy: %4.7f, cdf: %4.7f\n", i,
        p_energy_spectrum[i], p_cdf[i]);
    }
  }
}
