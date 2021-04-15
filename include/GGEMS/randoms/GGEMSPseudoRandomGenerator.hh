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
      \fn void Initialize(GGuint const& seed)
      \param seed - seed of the random
      \brief Initialize the Random object
    */
    void Initialize(GGuint const& seed);

    /*!
      \fn void SetSeed(GGuint const& seed)
      \param seed - seed of random
      \brief set the initial seed
    */
    void SetSeed(GGuint const& seed);

    /*!
      \fn void PrintInfos(void) const
      \brief printing infos about random
    */
    void PrintInfos(void) const;

    /*!
      \fn inline cl::Buffer* GetPseudoRandomNumbers(GGsize const& thread_index) const
      \param thread_index - index of activated device (thread index)
      \return pointer to OpenCL buffer storing random numbers
      \brief return the pointer to OpenCL buffer storing random numbers
    */
    inline cl::Buffer* GetPseudoRandomNumbers(GGsize const& thread_index) const {return pseudo_random_numbers_[thread_index];};

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

    /*!
      \fn GGuint GenerateSeed(void) const
      \return the seed computed by GGEMS
      \brief generate a seed by GGEMS and return it
    */
    GGuint GenerateSeed(void) const;

  private:
    cl::Buffer** pseudo_random_numbers_; /*!< Pointer storing the buffer about random numbers in activated device */
    GGsize number_activated_devices_; /*!< Number of activated device */
    GGuint seed_; /*!< Initial seed generating state of GGEMS random */
};

#endif // End of GUARD_GGEMS_RANDOMS_PSEUDO_RANDOM_GENERATOR_HH
