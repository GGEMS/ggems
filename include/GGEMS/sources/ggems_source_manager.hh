#ifndef GUARD_GGEMS_SOURCES_GGEMSSOURCEMANAGER_HH
#define GUARD_GGEMS_SOURCES_GGEMSSOURCEMANAGER_HH

/*!
  \file ggems_source_definition.hh

  \brief GGEMS class managing the source in GGEMS, every new sources in GGEMS
  inherit from this class

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday October 15, 2019
*/

#include "GGEMS/global/ggems_export.hh"
#include "GGEMS/global/opencl_manager.hh"
#include "GGEMS/tools/matrix.hh"

/*!
  \class GGEMSSourceManager
  \brief GGEMS class managing the source in GGEMS, every new sources in GGEMS
  inherit from this class
*/
class GGEMS_EXPORT GGEMSSourceManager
{
  public:
    /*!
      \fn static GGEMSSourceManager& GetInstance(void)
      \brief Create at first time the Singleton
      \return Object of type GGEMSSourceManager
    */
    static GGEMSSourceManager& GetInstance(void)
    {
      return *p_current_source_;
    }

  protected:
    /*!
      \brief GGEMSSourceManager constructor
    */
    GGEMSSourceManager(void);

    /*!
      \brief GGEMSSourceManager destructor
    */
    virtual ~GGEMSSourceManager(void);

  public:
    /*!
      \fn GGEMSSourceManager(GGEMSSourceManager const& ggems_source) = delete
      \param ggems_manager - reference on the ggems manager
      \brief Avoid copy of the class by reference
    */
    GGEMSSourceManager(GGEMSSourceManager const& ggems_source) = delete;

    /*!
      \fn GGEMSSourceManager& operator=(GGEMSSourceManager const& ggems_source) = delete
      \param ggems_manager - reference on the ggems manager
      \brief Avoid assignement of the class by reference
    */
    GGEMSSourceManager& operator=(
      GGEMSSourceManager const& ggems_source) = delete;

    /*!
      \fn GGEMSSourceManager(GGEMSSourceManager const&& ggems_source) = delete
      \param ggems_manager - rvalue reference on the ggems manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSSourceManager(GGEMSSourceManager const&& ggems_source) = delete;

    /*!
      \fn GGEMSSourceManager& operator=(GGEMSSourceManager const&& ggems_source) = delete
      \param ggems_manager - rvalue reference on the ggems manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSSourceManager& operator=(
      GGEMSSourceManager const&& ggems_source) = delete;

  public:
    /*!
      \fn bool IsReady(void) const
      \return return false is the source is not ready
      \brief Check if the source is ready to be used
    */
    bool IsReady(void) const;

    /*!
      \fn GGEMSSourceManager* GetSource() const
      \brief Get the pointer on the current source
      \return the pointer on the current source
    */
    inline GGEMSSourceManager* GetSource() const {return p_current_source_;};

    /*!
      \fn static void DeleteInstance(void)
      \brief Delete properly the singleton
    */
    static void DeleteInstance(void);

  public:
    /*!
      \fn void SetPosition(float const& pos_x, float const& pos_y, float const& pos_z)
      \param pos_x - Position of the source in X
      \param pos_y - Position of the source in Y
      \param pos_z - Position of the source in Z
      \brief Set the position of the source in the global coordinates
    */
    virtual void SetPosition(float const& pos_x, float const& pos_y,
      float const& pos_z);

    /*!
      \fn void SetParticleType(char const* particle_type)
      \param particle_type - Type of the particle
      \brief Set the type of the particle: electron, positron or photon
    */
    virtual void SetParticleType(char const* particle_type);

    /*!
      \fn void SetLocalAxis(float const& m00, float const& m01, float const& m02, float const& m10, float const& m11, float const& m12, float const& m20, float const& m21, float const& m22)
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
    virtual void SetLocalAxis(
      float const& m00, float const& m01, float const& m02,
      float const& m10, float const& m11, float const& m12,
      float const& m20, float const& m21, float const& m22);

  public: // Pure abstract method
    /*!
      \fn void GetPrimaries(cl::Buffer* p_primary_particles) = 0
      \param p_primary_particles - buffer of primary particles on OpenCL device
      \brief Generate primary particles
    */
    virtual void GetPrimaries(cl::Buffer* p_primary_particles) = 0;

    /*!
      \fn void Initialize(void) = 0
      \brief Initialize a GGEMS source
    */
    virtual void Initialize(void) = 0;

    /*!
      \fn void PrintInfos(void) const = 0
      \brief Printing infos about the source
    */
    virtual void PrintInfos(void) const = 0;

  private: // Pure abstract method
    /*!
      \fn void CheckParameters(void) const = 0
      \brief Check mandatory parameters for a source
    */
    virtual void CheckParameters(void) const = 0;

  protected:
    bool is_initialized_; /*!< Boolean checking if the source is initialized */
    cl_float3 source_position_; /*!< Position of the source */
    int particle_type_; /*!< Type of particle: photon, electron or positron */
    cl_float3 source_orbiting_angle_; /*!< Orbiting angle around the center of the world */
    Matrix::float3x3 local_axis_; /*!< 3x3 matrix for local axis */
    Matrix::float4x4 transformation_matrix_; /*!< Matrix transformation for local to global axis or global to local axis */

  private: // Storing the source
    static GGEMSSourceManager* p_current_source_; /*!< Current source */
};

#endif // End of GUARD_GGEMS_SOURCES_GGEMSSOURCEMANAGER_HH