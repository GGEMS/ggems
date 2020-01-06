#ifndef GUARD_GGEMS_SOURCES_GGEMSSOURCEMANAGER_HH
#define GUARD_GGEMS_SOURCES_GGEMSSOURCEMANAGER_HH

/*!
  \file GGEMSSourceManager.hh

  \brief GGEMS class managing the source in GGEMS, every new sources in GGEMS
  inherit from this class

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday October 15, 2019
*/

#ifdef _MSC_VER
#pragma warning(disable: 4251) // Deleting warning exporting STL members!!!
#endif

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/global/GGEMSOpenCLManager.hh"
#include "GGEMS/tools/GGEMSTypes.hh"

class GGEMSParticles;
class GGEMSPseudoRandomGenerator;
class GGEMSGeometryTransformation;

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
      \fn GGbool IsReady(void) const
      \return return false is the source is not ready
      \brief Check if the source is ready to be used
    */
    GGbool IsReady(void) const;

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
      \fn void SetPosition(GGfloat const& pos_x, GGfloat const& pos_y, GGfloat const& pos_z)
      \param pos_x - Position of the source in X
      \param pos_y - Position of the source in Y
      \param pos_z - Position of the source in Z
      \brief Set the position of the source in the global coordinates
    */
    void SetPosition(GGfloat const& pos_x, GGfloat const& pos_y,
      GGfloat const& pos_z);

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
    void SetLocalAxis(
      GGfloat const& m00, GGfloat const& m01, GGfloat const& m02,
      GGfloat const& m10, GGfloat const& m11, GGfloat const& m12,
      GGfloat const& m20, GGfloat const& m21, GGfloat const& m22);

    /*!
      \fn void SetRotation(GGfloat const& rx, GGfloat const& ry, GGfloat const& rz)
      \param rx - Rotation around X along global axis
      \param ry - Rotation around Y along global axis
      \param rz - Rotation around Z along global axis
      \brief Set the rotation of the source around global axis
    */
    void SetRotation(GGfloat const& rx, GGfloat const& ry, GGfloat const& rz);

    /*!
      \fn void UpdateRotation(GGfloat const& rx, GGfloat const& ry, GGfloat const& rz)
      \param rx - Rotation around X along global axis
      \param ry - Rotation around Y along global axis
      \param rz - Rotation around Z along global axis
      \brief Update the rotation of the source around global axis
    */
    void UpdateRotation(GGfloat const& rx, GGfloat const& ry,
      GGfloat const& rz);

    /*!
      \fn void SetParticle(GGEMSParticles* const p_particle)
      \param p_particle - pointer on particle
      \brief Set the particle pointer to source manager
    */
   void SetParticle(GGEMSParticles* const p_particle);

    /*!
      \fn void SetRandomGenerator(GGEMSPseudoRandomGenerator* const p_random_generator)
      \param p_random_generator - pointer on random generator
      \brief Set the random generator pointer to source manager
    */
   void SetRandomGenerator(
     GGEMSPseudoRandomGenerator* const p_random_generator);

  public: // Pure abstract method
    /*!
      \fn void GetPrimaries(GGulong const& number_of particles) = 0
      \param number_of_particles - number of particles to generate
      \brief Generate primary particles
    */
    virtual void GetPrimaries(GGulong const& number_of_particles) = 0;

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

  public:
    /*!
      \fn void CheckParameters(void) const
      \brief Check mandatory parameters for a source
    */
    virtual void CheckParameters(void) const;

  protected:
    bool is_initialized_; /*!< Boolean checking if the source is initialized */
    GGuchar particle_type_; /*!< Type of particle: photon, electron or positron */
    GGEMSGeometryTransformation* p_geometry_transformation_; /*!< Pointer storing the geometry transformation */

  protected: // kernel generating primaries
    cl::Kernel* p_kernel_get_primaries_; /*!< Kernel generating primaries on OpenCL device */

  protected: // Pointer on particle and random
    GGEMSParticles* p_particle_; /*!< Pointer storing infos about particles */
    GGEMSPseudoRandomGenerator* p_pseudo_random_generator_; /*!< Pointer storing infos about random numbers */

  protected:
    GGEMSOpenCLManager& opencl_manager_; /*!< Reference to opencl manager singleton */

  protected: // Storing the source
    inline static GGEMSSourceManager* p_current_source_ = nullptr; /*!< Current source */
};

#endif // End of GUARD_GGEMS_SOURCES_GGEMSSOURCEMANAGER_HH
