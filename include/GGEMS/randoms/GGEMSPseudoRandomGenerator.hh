#ifndef GUARD_GGEMS_RANDOMS_GGEMSPSEUDORANDOMGENERATOR_HH
#define GUARD_GGEMS_RANDOMS_GGEMSPSEUDORANDOMGENERATOR_HH

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
  \file GGEMSPseudoRandomGenerator.hh

  \brief Class managing the random number in GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday December 16, 2019
*/

#include "GGEMS/global/GGEMSConfiguration.hh"
#include "GGEMS/global/GGEMSOpenCLManager.hh"

/*!
  \class GGEMSPseudoRandomGenerator
  \brief Class managing the random number in GGEMS
*/
class GGEMS_EXPORT GGEMSPseudoRandomGenerator
{
  public:
    /*!
      \brief GGEMSPseudoRandomGenerator constructor
    */
    GGEMSPseudoRandomGenerator(void);

    /*!
      \brief GGEMSPseudoRandomGenerator destructor
    */
    ~GGEMSPseudoRandomGenerator(void);

  public:
    /*!
      \fn GGEMSPseudoRandomGenerator(GGEMSPseudoRandomGenerator const& random) = delete
      \param random - reference on the random
      \brief Avoid copy of the class by reference
    */
    GGEMSPseudoRandomGenerator(GGEMSPseudoRandomGenerator const& random) = delete;

    /*!
      \fn GGEMSPseudoRandomGenerator& operator=(GGEMSPseudoRandomGenerator const& random) = delete
      \param random - reference on the random
      \brief Avoid assignement of the class by reference
    */
    GGEMSPseudoRandomGenerator& operator=(GGEMSPseudoRandomGenerator const& random) = delete;

    /*!
      \fn GGEMSPseudoRandomGenerator(GGEMSPseudoRandomGenerator const&& random) = delete
      \param random - rvalue reference on the random
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSPseudoRandomGenerator(GGEMSPseudoRandomGenerator const&& random) = delete;

    /*!
      \fn GGEMSPseudoRandomGenerator& operator=(GGEMSPseudoRandomGenerator const&& random) = delete
      \param random - rvalue reference on the random
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSPseudoRandomGenerator& operator=(GGEMSPseudoRandomGenerator const&& random) = delete;

    /*!
      \fn void Initialize(void)
      \brief Initialize the Random object
    */
    void Initialize(void);

  public:
    /*!
      \fn inline cl::Buffer* GetPseudoRandomNumbers() const
      \return pointer to OpenCL buffer storing random numbers
      \brief return the pointer to OpenCL buffer storing random numbers
    */
    inline cl::Buffer* GetPseudoRandomNumbers() const {return pseudo_random_numbers_cl_.get();};

  private:
    /*!
      \fn void AllocateRandom(void)
      \brief Allocate memory for random numbers
    */
    void AllocateRandom(void);

    /*!
      \fn void InitializeSeeds(void)
      \brief Initialize seeds for random
    */
    void InitializeSeeds(void);

  private:
    std::shared_ptr<cl::Buffer> pseudo_random_numbers_cl_; /*!< Pointer storing the buffer about random numbers */
};

#endif // End of GUARD_GGEMS_RANDOMS_PSEUDO_RANDOM_GENERATOR_HH
