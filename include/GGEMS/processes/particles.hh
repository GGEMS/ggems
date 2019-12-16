#ifndef GUARD_GGEMS_PROCESSES_PARTICLES_HH
#define GUARD_GGEMS_PROCESSES_PARTICLES_HH

/*!
  \file particles.hh

  \brief Class managing the particles in GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Thrusday October 3, 2019
*/

#include "GGEMS/global/ggems_configuration.hh"
#include "GGEMS/global/ggems_export.hh"
#include "GGEMS/global/opencl_manager.hh"

/*!
  \class Particle
  \brief Class managing the particles in GGEMS
*/
class GGEMS_EXPORT Particle
{
  public:
    /*!
      \brief Particle constructor
    */
    Particle(void);

    /*!
      \brief Particle destructor
    */
    ~Particle(void);

  public:
    /*!
      \fn Particle(Particle const& particle) = delete
      \param ggems_manager - reference on the ggems manager
      \brief Avoid copy of the class by reference
    */
    Particle(Particle const& particle) = delete;

    /*!
      \fn Particle& operator=(Particle const& particle) = delete
      \param ggems_manager - reference on the ggems manager
      \brief Avoid assignement of the class by reference
    */
    Particle& operator=(Particle const& particle) = delete;

    /*!
      \fn Particle(Particle const&& particle) = delete
      \param ggems_manager - rvalue reference on the ggems manager
      \brief Avoid copy of the class by rvalue reference
    */
    Particle(Particle const&& particle) = delete;

    /*!
      \fn Particle& operator=(Particle const&& particle) = delete
      \param ggems_manager - rvalue reference on the ggems manager
      \brief Avoid copy of the class by rvalue reference
    */
    Particle& operator=(Particle const&& particle) = delete;

  public:
    /*!
      \fn void Initialize(void)
      \brief Initialize the Particle object
    */
    void Initialize(void);

  public:
    /*!
      \fn inline cl::Buffer* GetPrimaryParticles() const
      \return pointer to OpenCL buffer storing particles
      \brief return the pointer to OpenCL buffer storing particles
    */
    inline cl::Buffer* GetPrimaryParticles() const
    {
      return p_primary_particles_;
    };

  private:
    /*!
      \fn void AllocatePrimaryParticles(void)
      \brief Allocate memory for primary particles
    */
    void AllocatePrimaryParticles(void);

  private:
    cl::Buffer* p_primary_particles_; /*!< Pointer storing info about primary particles in batch on OpenCL device */
    OpenCLManager& opencl_manager_; /*!< Reference to OpenCL manager singleton */
};

#endif // End of GUARD_GGEMS_PROCESSES_PARTICLES_HH
