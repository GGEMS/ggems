#ifndef GUARD_GGEMS_RANDOMS_PSEUDO_RANDOM_GENERATOR_HH
#define GUARD_GGEMS_RANDOMS_PSEUDO_RANDOM_GENERATOR_HH

/*!
  \file pseudo_random_generator.hh

  \brief Class managing the random number in GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday December 16, 2019
*/

#include "GGEMS/global/ggems_configuration.hh"
#include "GGEMS/global/ggems_export.hh"
#include "GGEMS/global/opencl_manager.hh"

/*!
  \class RandomGenerator
  \brief Class managing the random number in GGEMS
*/
class GGEMS_EXPORT RandomGenerator
{
  public:
    /*!
      \brief RandomGenerator constructor
    */
    RandomGenerator(void);

    /*!
      \brief RandomGenerator destructor
    */
    ~RandomGenerator(void);

  public:
    /*!
      \fn RandomGenerator(RandomGenerator const& random) = delete
      \param ggems_manager - reference on the ggems manager
      \brief Avoid copy of the class by reference
    */
    RandomGenerator(RandomGenerator const& random) = delete;

    /*!
      \fn RandomGenerator& operator=(RandomGenerator const& random) = delete
      \param ggems_manager - reference on the ggems manager
      \brief Avoid assignement of the class by reference
    */
    RandomGenerator& operator=(RandomGenerator const& random) = delete;

    /*!
      \fn RandomGenerator(RandomGenerator const&& random) = delete
      \param ggems_manager - rvalue reference on the ggems manager
      \brief Avoid copy of the class by rvalue reference
    */
    RandomGenerator(RandomGenerator const&& random) = delete;

    /*!
      \fn RandomGenerator& operator=(RandomGenerator const&& random) = delete
      \param ggems_manager - rvalue reference on the ggems manager
      \brief Avoid copy of the class by rvalue reference
    */
    RandomGenerator& operator=(RandomGenerator const&& random) = delete;

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
      \fn inline cl::Buffer* GetRandomNumbers() const
      \return pointer to OpenCL buffer storing random numbers
      \brief return the pointer to OpenCL buffer storing random numbers
    */
    inline cl::Buffer* GetRandomNumbers() const {return p_random_numbers_;};

  private:
    cl::Buffer* p_random_numbers_; /*!< Pointer storing the buffer about random numbers */
    OpenCLManager& opencl_manager_; /*!< Reference to OpenCL manager singleton */
};

#endif // End of GUARD_GGEMS_RANDOMS_PSEUDO_RANDOM_GENERATOR_HH
