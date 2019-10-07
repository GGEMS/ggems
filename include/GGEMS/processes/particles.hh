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

#include "GGEMS/global/ggems_export.hh"
#include "GGEMS/global/opencl_manager.hh"

/*!
  \struct PrimaryParticles_t
  \brief Structure storing informations about primary particles for OpenCL
*/
typedef struct GGEMS_EXPORT PrimaryParticles_t
{
  cl::Buffer* p_E_; /*!< Energies of particles in float */
  cl::Buffer* p_dx_; /*!< Position of the particle in x in float */
  cl::Buffer* p_dy_; /*!< Position of the particle in y in float */
  cl::Buffer* p_dz_; /*!< Position of the particle in z in float */
  cl::Buffer* p_px_; /*!< Momentum of the particle in x in float */
  cl::Buffer* p_py_; /*!< Momentum of the particle in y in float */
  cl::Buffer* p_pz_; /*!< Momentum of the particle in z in float */
  cl::Buffer* tof_; /*!< Time of flight in float */

  cl::Buffer* p_prng_state_1_; /*!< State 1 of the prng in unsigned int 32 */
  cl::Buffer* p_prng_state_2_; /*!< State 2 of the prng in unsigned int 32 */
  cl::Buffer* p_prng_state_3_; /*!< State 3 of the prng in unsigned int 32 */
  cl::Buffer* p_prng_state_4_; /*!< State 4 of the prng in unsigned int 32 */
  cl::Buffer* p_prng_state_5_; /*!< State 5 of the prng in unsigned int 32 */

  cl::Buffer* p_geometry_id_; /*!< current geometry crossed by the particle in unsigned int 32 */
  cl::Buffer* p_E_index_; /*!< Energy index within CS and Mat tables in unsigned int 16 */
  cl::Buffer* p_scatter_order_; /*!< Scatter order, usefull for the imageryin unsigned int 16 */

  cl::Buffer* p_next_interaction_distance_; /*!< Distance to the next interaction in float */
  cl::Buffer* p_next_discrete_process_; /*!< Next process in unsigned int 8 */

  cl::Buffer* p_status_; /*!< in unsigned int 8 */
  cl::Buffer* p_level_; /*!<  in unsigned int 8 */
  cl::Buffer* p_pname_; /*!< particle name (photon, electron, etc) in unsigned int 8 */

  cl_ulong number_of_primaries_; /*!< Number of the primaries */
} PrimaryParticles;

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

  private:
    /*!
      \fn void AllocatePrimaryParticles()
      \brief Allocate memory for primary particles
    */
    void AllocatePrimaryParticles();

  private:
    uint64_t number_of_particles_; /*!< Number of the particles to simulate in a batch */
    PrimaryParticles* p_primary_particles_; /*!< Pointer storing info about primary particles in batch */
};

#endif // End of GUARD_GGEMS_PROCESSES_PARTICLES_HH
