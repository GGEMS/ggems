__kernel void print_primary_particle(__global float* p_E,
  int const index)
{
  // Get the index of thread
  int const kGlobalIndex = get_global_id(0);

  if (kGlobalIndex == index) {
    printf("Index: %d, float: %4.7f\n", kGlobalIndex, p_E[kGlobalIndex]);
  }
}
