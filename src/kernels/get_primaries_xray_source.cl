//#include "GGEMS/auxiliary/use_double_precision.hh"

__kernel void get_primaries_xray_source(
  //__global double const* p_cdf,
  //__global double const* p_energy_spectrum,
  __global float const* p_test,
  unsigned int const number_of_energy_bins)
{
  // Get the index of thread
  int const kGlobalIndex = get_global_id(0);

  if (kGlobalIndex == 2) {
    printf("Particle id: %d\n", kGlobalIndex);
    printf("Number of energy bin: %u\n", number_of_energy_bins);
    for (unsigned int i = 0; i < number_of_energy_bins; ++i) {
      printf("Energy bin: %u, energy: %4.7f\n", i, p_test[i]);
    }
  }
}
