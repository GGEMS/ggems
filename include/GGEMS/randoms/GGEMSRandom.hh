#ifndef GUARD_GGEMS_RANDOMS_GGEMSRANDOMSTACK_HH
#define GUARD_GGEMS_RANDOMS_GGEMSRANDOMSTACK_HH

// ************************************************************************
// * This file is part of GGEMS.                                          *
// *                                                                      *
// * GGEMS is free software: you can redistribute it and/or modify        *
// * it under the terms of the GNU General Public License as published by *
// * the Free Software Foundation, either version 3 of the License, or    *
// * (at your option) any later version.                                  *
// *                                                                      *
// * GGEMS is distributed in the hope that it will be useful,             *
// * but WITHOUT ANY WARRANTY; without even the implied warranty of       *
// * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        *
// * GNU General Public License for more details.                         *
// *                                                                      *
// * You should have received a copy of the GNU General Public License    *
// * along with GGEMS.  If not, see <https://www.gnu.org/licenses/>.      *
// *                                                                      *
// ************************************************************************

/*!
  \file GGEMSRandomStack.hh

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
  \struct GGEMSRandom_t
  \brief Structure storing informations about random
*/
#pragma pack(push, 1)
typedef struct GGEMSRandom_t
{
  GGuint prng_state_1_[MAXIMUM_PARTICLES]; /*!< State 1 of the prng */
  GGuint prng_state_2_[MAXIMUM_PARTICLES]; /*!< State 2 of the prng */
  GGuint prng_state_3_[MAXIMUM_PARTICLES]; /*!< State 3 of the prng */
  GGuint prng_state_4_[MAXIMUM_PARTICLES]; /*!< State 4 of the prng */
  GGuint prng_state_5_[MAXIMUM_PARTICLES]; /*!< State 5 of the prng */
} GGEMSRandom; /*!< Using C convention name of struct to C++ (_t deletion) */
#pragma pack(pop)

#endif // End of GUARD_GGEMS_RANDOMS_GGEMSRANDOMSTACK_HH
