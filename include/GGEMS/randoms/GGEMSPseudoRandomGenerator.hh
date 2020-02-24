#ifndef GUARD_GGEMS_RANDOMS_GGEMSPSEUDORANDOMGENERATOR_HH
#define GUARD_GGEMS_RANDOMS_GGEMSPSEUDORANDOMGENERATOR_HH

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
#include "GGEMS/global/GGEMSExport.hh"
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
      \param ggems_manager - reference on the ggems manager
      \brief Avoid copy of the class by reference
    */
    GGEMSPseudoRandomGenerator(GGEMSPseudoRandomGenerator const& random) = delete;

    /*!
      \fn GGEMSPseudoRandomGenerator& operator=(GGEMSPseudoRandomGenerator const& random) = delete
      \param ggems_manager - reference on the ggems manager
      \brief Avoid assignement of the class by reference
    */
    GGEMSPseudoRandomGenerator& operator=(GGEMSPseudoRandomGenerator const& random) = delete;

    /*!
      \fn GGEMSPseudoRandomGenerator(GGEMSPseudoRandomGenerator const&& random) = delete
      \param ggems_manager - rvalue reference on the ggems manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSPseudoRandomGenerator(GGEMSPseudoRandomGenerator const&& random) = delete;

    /*!
      \fn GGEMSPseudoRandomGenerator& operator=(GGEMSPseudoRandomGenerator const&& random) = delete
      \param ggems_manager - rvalue reference on the ggems manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSPseudoRandomGenerator& operator=(GGEMSPseudoRandomGenerator const&& random) = delete;

  public:
    /*!
      \fn void Initialize(void)
      \brief Initialize the Random object
    */
    void Initialize(void);

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

  public:
    /*!
      \fn inline cl::Buffer* GetPseudoRandomNumbers() const
      \return pointer to OpenCL buffer storing random numbers
      \brief return the pointer to OpenCL buffer storing random numbers
    */
    inline cl::Buffer* GetPseudoRandomNumbers() const {return pseudo_random_numbers_.get();};

  private:
    std::shared_ptr<cl::Buffer> pseudo_random_numbers_; /*!< Pointer storing the buffer about random numbers */
    GGEMSOpenCLManager& opencl_manager_; /*!< Reference to OpenCL manager singleton */
};

#endif // End of GUARD_GGEMS_RANDOMS_PSEUDO_RANDOM_GENERATOR_HH
