#ifndef GUARD_GGEMS_SOURCES_GGEMSSOURCE_HH
#define GUARD_GGEMS_SOURCES_GGEMSSOURCE_HH

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
#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSTypes.hh"

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
    explicit GGEMSSource(GGEMSSource* source);

    /*!
      \brief GGEMSSource destructor
    */
    virtual ~GGEMSSource(void);

  public:
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
      \fn void SetSourceName(char const* source_name)
      \param source_name - name of the source
      \brief save the name of the source
    */
    void SetSourceName(char const* source_name);

    /*!
      \fn void SetPosition(GGfloat const& pos_x, GGfloat const& pos_y, GGfloat const& pos_z, char const* unit = "mm")
      \param pos_x - Position of the source in X
      \param pos_y - Position of the source in Y
      \param pos_z - Position of the source in Z
      \param unit - unit of the distance
      \brief Set the position of the source in the global coordinates
    */
    void SetPosition(GGfloat const& pos_x, GGfloat const& pos_y, GGfloat const& pos_z, char const* unit = "mm");

    /*!
      \fn void SetSourceParticleType(char const* particle_type)
      \param particle_type - Type of the particle
      \brief Set the type of the particle: electron, positron or photon
    */
    void SetSourceParticleType(char const* particle_type);

    /*!
      \fn void SetLocalAxis(GGfloat const& m00, GGfloat const& m01, GGfloat const& m02, GGfloat const& m10, GGfloat const& m11, GGfloat const& m12, GGfloat const& m20, GGfloat const& m21, GGfloat const& m22)
      \param m00 - Element 0,0 in the matrix 3x3 for local axis
      \param m01 - Element 0,1 in the matrix 3x3 for local axis
      \param m02 - Element 0,2 in the matrix 3x3 for local axis
      \param m10 - Element 1,0 in the matrix 3x3 for local axis
      \param m11 - Element 1,1 in the matrix 3x3 for local axis
      \param m12 - Element 1,2 in the matrix 3x3 for local axis
      \param m20 - Element 2,0 in the matrix 3x3 for local axis
      \param m21 - Element 2,1 in the matrix 3x3 for local axis
      \param m22 - Element 2,2 in the matrix 3x3 for local axis
      \brief Set the local axis element describing the source compared to global axis (center of world)
    */
    void SetLocalAxis(GGfloat const& m00, GGfloat const& m01, GGfloat const& m02, GGfloat const& m10, GGfloat const& m11, GGfloat const& m12, GGfloat const& m20, GGfloat const& m21, GGfloat const& m22);

    /*!
      \fn void SetRotation(GGfloat const& rx, GGfloat const& ry, GGfloat const& rz)
      \param rx - Rotation around X along global axis
      \param ry - Rotation around Y along global axis
      \param rz - Rotation around Z along global axis
      \param unit - unit of the angle
      \brief Set the rotation of the source around global axis
    */
    void SetRotation(GGfloat const& rx, GGfloat const& ry, GGfloat const& rz, char const* unit = "deg");

    /*!
      \fn void SetNumberOfParticles(GGulong const& number_of_particles)
      \param number_of_particles - number of particles to simulate
      \brief Set the number of particles to simulate during the simulation
    */
    void SetNumberOfParticles(GGulong const& number_of_particles);

    /*!
      \fn inline std::size_t GetNumberOfBatchs(void) const
      \return the number of batch of particle
      \brief method returning the number of particles by batch
    */
    inline std::size_t GetNumberOfBatchs(void) const {return number_of_particles_in_batch_.size();}

    /*!
      \fn inline GGulong GetNumberOfParticlesInBatch(std::size_t const& batch_index)
      \return the number of particle for a specific batch
      \brief method returning the number of particles in a specific batch
    */
    inline GGulong GetNumberOfParticlesInBatch(std::size_t const& batch_index) {return number_of_particles_in_batch_.at(batch_index);}

    /*!
      \fn void GetPrimaries(GGulong const& number_of particles) = 0
      \param number_of_particles - number of particles to generate
      \brief Generate primary particles
    */
    virtual void GetPrimaries(GGulong const& number_of_particles) = 0;

    /*!
      \fn void PrintInfos(void) const = 0
      \brief Printing infos about the source
    */
    virtual void PrintInfos(void) const = 0;

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

    /*!
      \fn void CheckMemoryForParticles(void) const
      \brief Check the memory for particles and propose an optimized MAXIMUM_NUMBER of particles if necessary
    */
    void CheckMemoryForParticles(void) const;

  protected:
    std::string source_name_; /*!< Name of the source */
    GGulong number_of_particles_; /*!< Number of particles */
    std::vector<unsigned long long> number_of_particles_in_batch_; /*!< Number of particles in batch */
    GGuchar particle_type_; /*!< Type of particle: photon, electron or positron */
    std::unique_ptr<GGEMSGeometryTransformation> geometry_transformation_; /*!< Pointer storing the geometry transformation */
    std::shared_ptr<cl::Kernel> kernel_get_primaries_; /*!< Kernel generating primaries on OpenCL device */
    GGEMSOpenCLManager& opencl_manager_; /*!< Reference to opencl manager singleton */
};

#endif // End of GUARD_GGEMS_SOURCES_GGEMSSOURCE_HH
