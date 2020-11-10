#ifndef GUARD_GGEMS_SOURCES_GGEMSSOURCE_HH
#define GUARD_GGEMS_SOURCES_GGEMSSOURCE_HH

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
  \file GGEMSSource.hh

  \brief GGEMS mother class for the source

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday October 15, 2019
*/

#include "GGEMS/global/GGEMSOpenCLManager.hh"

class GGEMSParticles;
class GGEMSPseudoRandomGenerator;
class GGEMSGeometryTransformation;

/*!
  \class GGEMSSource
  \brief GGEMS mother class for the source
*/
class GGEMS_EXPORT GGEMSSource
{
  public:
    /*!
      \brief GGEMSSource constructor
    */
    explicit GGEMSSource(std::string const& source_name);

    /*!
      \brief GGEMSSource destructor
    */
    virtual ~GGEMSSource(void);

    /*!
      \fn GGEMSSource(GGEMSSource const& source) = delete
      \param source - reference on the GGEMS source
      \brief Avoid copy by reference
    */
    GGEMSSource(GGEMSSource const& source) = delete;

    /*!
      \fn GGEMSSource& operator=(GGEMSSource const& source) = delete
      \param source - reference on the GGEMS source
      \brief Avoid assignement by reference
    */
    GGEMSSource& operator=(GGEMSSource const& source) = delete;

    /*!
      \fn GGEMSSource(GGEMSSource const&& source) = delete
      \param source - rvalue reference on the GGEMS source
      \brief Avoid copy by rvalue reference
    */
    GGEMSSource(GGEMSSource const&& source) = delete;

    /*!
      \fn GGEMSSource& operator=(GGEMSSource const&& source) = delete
      \param source - rvalue reference on the GGEMS source
      \brief Avoid copy by rvalue reference
    */
    GGEMSSource& operator=(GGEMSSource const&& source) = delete;

    /*!
      \fn inline std::string GetNameOfSource(void) const
      \return name of the source
      \brief get the name of the source
    */
    inline std::string GetNameOfSource(void) const {return source_name_;}

    /*!
      \fn void SetPosition(GGfloat const& pos_x, GGfloat const& pos_y, GGfloat const& pos_z, std::string const& unit = "mm")
      \param pos_x - Position of the source in X
      \param pos_y - Position of the source in Y
      \param pos_z - Position of the source in Z
      \param unit - unit of the distance
      \brief Set the position of the source in the global coordinates
    */
    void SetPosition(GGfloat const& pos_x, GGfloat const& pos_y, GGfloat const& pos_z, std::string const& unit = "mm");

    /*!
      \fn void SetSourceParticleType(std::string const& particle_type)
      \param particle_type - Type of the particle
      \brief Set the type of the particle: electron, positron or photon
    */
    void SetSourceParticleType(std::string const& particle_type);

    /*!
      \fn void SetRotation(GGfloat const& rx, GGfloat const& ry, GGfloat const& rz, std::string const& unit)
      \param rx - Rotation around X along global axis
      \param ry - Rotation around Y along global axis
      \param rz - Rotation around Z along global axis
      \param unit - unit of the angle
      \brief Set the rotation of the source around global axis
    */
    void SetRotation(GGfloat const& rx, GGfloat const& ry, GGfloat const& rz, std::string const& unit = "deg");

    /*!
      \fn void SetNumberOfParticles(GGlong const& number_of_particles)
      \param number_of_particles - number of particles to simulate
      \brief Set the number of particles to simulate during the simulation
    */
    void SetNumberOfParticles(GGlong const& number_of_particles);

    /*!
      \fn void EnableTracking(void)
      \brief Enabling tracking infos during simulation
    */
    void EnableTracking(void);

    /*!
      \fn inline std::size_t GetNumberOfBatchs(void) const
      \return the number of batch of particle
      \brief method returning the number of particles by batch
    */
    inline std::size_t GetNumberOfBatchs(void) const {return number_of_particles_in_batch_.size();}

    /*!
      \fn inline GGulong GetNumberOfParticlesInBatch(std::size_t const& batch_index)
      \param batch_index - index of the batch
      \return the number of particle for a specific batch
      \brief method returning the number of particles in a specific batch
    */
    inline GGlong GetNumberOfParticlesInBatch(std::size_t const& batch_index) {return number_of_particles_in_batch_.at(batch_index);}

    /*!
      \fn void CheckParameters(void) const
      \brief Check mandatory parameters for a source
    */
    virtual void CheckParameters(void) const;

    /*!
      \fn void Initialize(void)
      \brief Initialize a GGEMS source
    */
    virtual void Initialize(void);

    /*!
      \fn void GetPrimaries(GGlong const& number_of particles) = 0
      \param number_of_particles - number of particles to generate
      \brief Generate primary particles
    */
    virtual void GetPrimaries(GGlong const& number_of_particles) = 0;

    /*!
      \fn void PrintInfos(void) const = 0
      \brief Printing infos about the source
    */
    virtual void PrintInfos(void) const = 0;

  protected:
    /*!
      \fn void InitializeKernel(void)
      \brief Initialize kernel for specific source in OpenCL
    */
    virtual void InitializeKernel(void) = 0;

  private:
    /*!
      \fn void OrganizeParticlesInBatch
      \brief Organize the particles in batch
    */
    void OrganizeParticlesInBatch(void);

  protected:
    std::string source_name_; /*!< Name of the source */
    GGlong number_of_particles_; /*!< Number of particles */
    std::vector<GGlong> number_of_particles_in_batch_; /*!< Number of particles in batch */
    GGchar particle_type_; /*!< Type of particle: photon, electron or positron */
    std::string tracking_kernel_option_; /*!< Preprocessor option for tracking */
    std::unique_ptr<GGEMSGeometryTransformation> geometry_transformation_; /*!< Pointer storing the geometry transformation */
    std::weak_ptr<cl::Kernel> kernel_get_primaries_cl_; /*!< Kernel generating primaries on OpenCL device */
};

#endif // End of GUARD_GGEMS_SOURCES_GGEMSSOURCE_HH
