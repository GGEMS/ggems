#ifndef GUARD_GGEMS_RANDOMS_GGEMSRANDOMSTACK_HH
#define GUARD_GGEMS_RANDOMS_GGEMSRANDOMSTACK_HH

/*!
  \file random.hh

  \brief Structure storing the random buffers for both OpenCL and GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday December 16, 2019
*/

#include "GGEMS/global/GGEMSConfiguration.hh"
#include "GGEMS/tools/GGEMSTypes.hh"

/*!
  \struct Random_t
  \brief Structure storing informations about random
*/
#ifdef OPENCL_COMPILER
typedef struct __attribute__((aligned (1))) Random_t
#else
#ifdef _MSC_VER
#pragma pack(push, 1)
#endif
typedef struct PACKED Random_t
#endif
{
  GGuint p_prng_state_1_[MAXIMUM_PARTICLES]; /*!< State 1 of the prng */
  GGuint p_prng_state_2_[MAXIMUM_PARTICLES]; /*!< State 2 of the prng */
  GGuint p_prng_state_3_[MAXIMUM_PARTICLES]; /*!< State 3 of the prng */
  GGuint p_prng_state_4_[MAXIMUM_PARTICLES]; /*!< State 4 of the prng */
  GGuint p_prng_state_5_[MAXIMUM_PARTICLES]; /*!< State 5 of the prng */
} Random;
#ifndef OPENCL_COMPILER
#ifdef _MSC_VER
#pragma pack(pop)
#endif
#endif

#endif // End of GUARD_GGEMS_RANDOMS_GGEMSRANDOMSTACK_HH
