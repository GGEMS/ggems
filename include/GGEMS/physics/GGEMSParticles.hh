#ifndef GUARD_GGEMS_PHYSICS_GGEMSPARTICLES_HH
#define GUARD_GGEMS_PHYSICS_GGEMSPARTICLES_HH

/*!
  \file GGEMSParticles.hh

  \brief Class managing the particles in GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Thrusday October 3, 2019
*/

#include "GGEMS/global/GGEMSConfiguration.hh"
#include "GGEMS/global/GGEMSOpenCLManager.hh"
#include "GGEMS/tools/GGEMSSystemOfUnits.hh"
#include "GGEMS/physics/GGEMSParticleConstants.hh"

/*!
  \class GGEMSParticles
  \brief Class managing the particles in GGEMS
*/
class GGEMS_EXPORT GGEMSParticles
{
  public:
    /*!
      \brief GGEMSParticles constructor
    */
    GGEMSParticles(void);

    /*!
      \brief GGEMSParticles destructor
    */
    ~GGEMSParticles(void);

    /*!
      \fn GGEMSParticles(GGEMSParticles const& particle) = delete
      \param particle - reference on the particle
      \brief Avoid copy of the class by reference
    */
    GGEMSParticles(GGEMSParticles const& particle) = delete;

    /*!
      \fn GGEMSParticles& operator=(GGEMSParticles const& particle) = delete
      \param particle - reference on the particle
      \brief Avoid assignement of the class by reference
    */
    GGEMSParticles& operator=(GGEMSParticles const& particle) = delete;

    /*!
      \fn GGEMSParticles(GGEMSParticles const&& particle) = delete
      \param particle - rvalue reference on the particle
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSParticles(GGEMSParticles const&& particle) = delete;

    /*!
      \fn GGEMSParticles& operator=(GGEMSParticles const&& particle) = delete
      \param particle - rvalue reference on the particle
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSParticles& operator=(GGEMSParticles const&& particle) = delete;

    /*!
      \fn void Initialize(void)
      \brief Initialize the GGEMSParticles object
    */
    void Initialize(void);

    /*!
      \fn inline cl::Buffer* GetPrimaryParticles() const
      \return pointer to OpenCL buffer storing particles
      \brief return the pointer to OpenCL buffer storing particles
    */
    inline cl::Buffer* GetPrimaryParticles() const {return primary_particles_.get();};

    /*!
      \fn void SetNumberOfParticles(GGulong const& number_of_particles)
      \param number_of_particles - number of activated particles in buffer
      \brief Set the number of particles in buffer
    */
    void SetNumberOfParticles(GGulong const& number_of_particles);

    /*!
      \fn inline GGulong GetNumberOfParticles(void) const
      \return number of particles currently activated in OpenCL buffer
      \brief Get the number of particles
    */
    inline GGulong GetNumberOfParticles(void) const {return number_of_particles_;};

  private:
    /*!
      \fn void AllocatePrimaryParticles(void)
      \brief Allocate memory for primary particles
    */
    void AllocatePrimaryParticles(void);

  private:
    GGulong number_of_particles_; /*!< Number of activated particles in buffer */
    std::shared_ptr<cl::Buffer> primary_particles_; /*!< Pointer storing info about primary particles in batch on OpenCL device */
};

#endif // End of GUARD_GGEMS_PHYSICS_GGEMSPARTICLES_HH
