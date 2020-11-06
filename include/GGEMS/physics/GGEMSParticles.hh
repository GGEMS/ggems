#ifndef GUARD_GGEMS_PHYSICS_GGEMSPARTICLES_HH
#define GUARD_GGEMS_PHYSICS_GGEMSPARTICLES_HH

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
  \file GGEMSParticles.hh

  \brief Class managing the particles in GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Thrusday October 3, 2019
*/

#include <map>

#include "GGEMS/global/GGEMSConfiguration.hh"
#include "GGEMS/global/GGEMSOpenCLManager.hh"
#include "GGEMS/tools/GGEMSSystemOfUnits.hh"
#include "GGEMS/physics/GGEMSParticleConstants.hh"

typedef std::map<GGchar, std::string> ParticleTypeMap; /*!< Map with key: particle index, element: name of particle */
typedef std::map<GGchar, std::string> ParticleStatusMap; /*!< Map with key: particle index, element: status */
typedef std::map<GGchar, std::string> ParticleLevelMap; /*!< Map with key: particle index, element: level (primary or secondary) */

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
    inline cl::Buffer* GetPrimaryParticles() const {return primary_particles_cl_.get();};

    /*!
      \fn void SetNumberOfParticles(GGlong const& number_of_particles)
      \param number_of_particles - number of activated particles in buffer
      \brief Set the number of particles in buffer
    */
    void SetNumberOfParticles(GGlong const& number_of_particles);

    /*!
      \fn inline GGlong GetNumberOfParticles(void) const
      \return number of particles currently activated in OpenCL buffer
      \brief Get the number of particles
    */
    inline GGlong GetNumberOfParticles(void) const {return number_of_particles_;};

    /*!
      \fn bool IsAlive(void) const
      \return true if source is still alive, otherwize false
      \brief check if some particles are alive in OpenCL particle buffer
    */
    bool IsAlive(void) const;

    /*!
      \fn void Dump(std::string const& message) const
      \param message - message for dumping
      \brief dump particle infos
    */
    void Dump(std::string const& message) const;

  private:
    /*!
      \fn void AllocatePrimaryParticles(void)
      \brief Allocate memory for primary particles
    */
    void AllocatePrimaryParticles(void);

  private:
    GGlong number_of_particles_; /*!< Number of activated particles in buffer */
    std::shared_ptr<cl::Buffer> primary_particles_cl_; /*!< Pointer storing info about primary particles in batch on OpenCL device */
    ParticleTypeMap particle_type_; /*!< Map of particle type */
    ParticleStatusMap particle_status_; /*!< Map of particle status */
    ParticleLevelMap particle_level_; /*!< Map of particle level */
};

#endif // End of GUARD_GGEMS_PHYSICS_GGEMSPARTICLES_HH
