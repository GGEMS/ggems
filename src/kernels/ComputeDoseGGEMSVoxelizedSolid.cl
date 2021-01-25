#include "GGEMS/tools/GGEMSTypes.hh"
#include "GGEMS/navigators/GGEMSDoseParams.hh"
#include "GGEMS/materials/GGEMSMaterialTables.hh"

/*!
  \fn kernel void compute_dose_ggems_voxelized_solid(global GGEMSDoseParams const* dose_params, global GGDosiType const* edep, , global GGshort const* label_data, global GGEMSMaterialTables const* materials, global GGfloat* dose)
  \param dose_params -
  \param edep -
  \param label_data -
  \param materials -
  \param dose -
  \brief computing dose for voxelized solid
*/
kernel void compute_dose_ggems_voxelized_solid(global GGEMSDoseParams const* dose_params, global GGDosiType const* edep, , global GGshort const* label_data, global GGEMSMaterialTables const* materials, global GGfloat* dose)
{
  // Getting index of thread
  GGint global_id = get_global_id(0);
}
