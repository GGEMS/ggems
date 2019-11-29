#include "GGEMS/auxiliary/use_double_precision.hh"

__kernel void get_primaries_xray_source()
{
  // Get the index of thread
  int const kGlobalIndex = get_global_id(0);
}
