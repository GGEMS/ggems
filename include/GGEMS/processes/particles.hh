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
  \struct PrimaryParticles_t
  \brief Structure storing informations about primary particles for OpenCL
*/
#if defined _MSC_VER
#pragma pack(push, 1)
#endif
typedef struct PACKED PrimaryParticles_t
{
  cl_ulong number_of_primaries_; /*!< Number of the primaries */
  
  cl_float p_E_[MAXIMUM_PARTICLES]; /*!< Energies of particles */
  cl_float p_dx_[MAXIMUM_PARTICLES]; /*!< Position of the particle in x */
  cl_float p_dy_[MAXIMUM_PARTICLES]; /*!< Position of the particle in y */
  cl_float p_dz_[MAXIMUM_PARTICLES]; /*!< Position of the particle in z */
  cl_float p_px_[MAXIMUM_PARTICLES]; /*!< Momentum of the particle in x */
  cl_float p_py_[MAXIMUM_PARTICLES]; /*!< Momentum of the particle in y */
  cl_float p_pz_[MAXIMUM_PARTICLES]; /*!< Momentum of the particle in z */
  cl_float p_tof_[MAXIMUM_PARTICLES]; /*!< Time of flight */

  cl_uint p_prng_state_1_[MAXIMUM_PARTICLES]; /*!< State 1 of the prng */
  cl_uint p_prng_state_2_[MAXIMUM_PARTICLES]; /*!< State 2 of the prng */
  cl_uint p_prng_state_3_[MAXIMUM_PARTICLES]; /*!< State 3 of the prng */
  cl_uint p_prng_state_4_[MAXIMUM_PARTICLES]; /*!< State 4 of the prng */
  cl_uint p_prng_state_5_[MAXIMUM_PARTICLES]; /*!< State 5 of the prng */

  cl_uint p_geometry_id_[MAXIMUM_PARTICLES]; /*!< current geometry crossed by the particle */
  cl_ushort p_E_index_[MAXIMUM_PARTICLES]; /*!< Energy index within CS and Mat tables */
  cl_ushort p_scatter_order_[MAXIMUM_PARTICLES]; /*!< Scatter order, usefull for the imagery */

  cl_float p_next_interaction_distance_[MAXIMUM_PARTICLES]; /*!< Distance to the next interaction */
  cl_uchar p_next_discrete_process_[MAXIMUM_PARTICLES]; /*!< Next process */

  cl_uchar p_status_[MAXIMUM_PARTICLES]; /*!< */
  cl_uchar p_level_[MAXIMUM_PARTICLES]; /*!< */
  cl_uchar* p_pname_[MAXIMUM_PARTICLES]; /*!< particle name (photon, electron, etc) */

  
} PrimaryParticles;
#if defined _MSC_VER
#pragma pack(pop)
#endif

/*!
  \struct PrimaryParticles
  \brief Structure storing informations about secondary particles
*/
typedef struct  GGEMS_EXPORT SecondaryParticles_t
{
  
} SecondaryParticles;

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
    Particle();

    /*!
      \brief Particle destructor
    */
    ~Particle();

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
      \fn void Initialize()
      \brief Initialize the Particle object
    */
    void Initialize();

    /*!
      \fn void SetNumberOfParticlesInBatch(uint64_t const& number_of_particles_in_batch)
      \param number_of_particles_in_batch - Number of the particles in the batch
      \brief Set the number of the particles to simulate in a batch
    */
    void SetNumberOfParticlesInBatch(
      uint64_t const& number_of_particles_in_batch);

    /*!
      \fn PrimaryParticles* GetPrimaryParticlesDevice() const
      \brief Get the pointer on primary particles on OpenCL device memory
      \return The pointer on primary particles in device memory
    */
    PrimaryParticles* GetPrimaryParticlesDevice() const;

    void ReleasePrimaryParticlesDevice(
      PrimaryParticles* p_primary_particles) const;

  private:
    /*!
      \fn void AllocatePrimaryParticles()
      \brief Allocate memory for primary particles
    */
    void AllocatePrimaryParticles();

    /*!
      \fn void InitializeSeeds()
      \brief Initialize seeds for each particle
    */
    void InitializeSeeds();

  private:
    uint64_t number_of_particles_; /*!< Number of the particles to simulate in a batch */
    //PrimaryParticles* p_primary_particles_; /*!< Pointer storing info about primary particles in batch */
    cl::Buffer* p_primary_particles_;
};

#endif // End of GUARD_GGEMS_PROCESSES_PARTICLES_HH
