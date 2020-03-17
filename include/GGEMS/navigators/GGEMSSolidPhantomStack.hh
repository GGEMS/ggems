#ifndef GUARD_GGEMS_NAVIGATORS_GGEMSSOLIDPHANTOMSTACK_HH
#define GUARD_GGEMS_NAVIGATORS_GGEMSSOLIDPHANTOMSTACK_HH

/*!
  \file GGEMSSolidPhantomStack.hh

  \brief Structure storing the stack of data for solid phantom

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday March 2, 2020
*/

#include "GGEMS/global/GGEMSConfiguration.hh"
#include "GGEMS/tools/GGEMSTypes.hh"

/*!
  \struct GGEMSSolidPhantomData_t
  \brief Structure storing the stack of data for solid phantom
*/
#ifdef OPENCL_COMPILER
typedef struct __attribute__((aligned (1))) GGEMSSolidPhantomData_t
#else
#ifdef _MSC_VER
#pragma pack(push, 1)
#endif
typedef struct PACKED GGEMSSolidPhantomData_t
#endif
{
  GGushort3 number_of_voxels_xyz_; /*!< Number of voxel in X, Y and Z [0, 65535] */
  GGuint number_of_voxels_; /*!< Total number of voxels */
  GGdouble3 voxel_sizes_xyz_; /*!< Size of voxels in X, Y and Z */
  GGdouble3 offsets_xyz_; /*!< Offset of phantom in X, Y and Z */
  GGdouble3 border_min_xyz_; /*!< Min. of border in X, Y and Z */
  GGdouble3 border_max_xyz_; /*!< Max. of border in X, Y and Z */
} GGEMSSolidPhantomData;
#ifndef OPENCL_COMPILER
#ifdef _MSC_VER
#pragma pack(pop)
#endif
#endif

#endif // GUARD_GGEMS_NAVIGATORS_GGEMSSOLIDPHANTOMSTACK_HH
