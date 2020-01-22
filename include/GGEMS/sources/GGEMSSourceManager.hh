#ifndef GUARD_GGEMS_SOURCES_GGEMSSOURCEMANAGER_HH
#define GUARD_GGEMS_SOURCES_GGEMSSOURCEMANAGER_HH

/*!
  \file GGEMSSourceManager.hh

  \brief GGEMS class handling the source(s)

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Thursday January 16, 2020
*/

#ifdef _MSC_VER
#pragma warning(disable: 4251) // Deleting warning exporting STL members!!!
#endif

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSTypes.hh"
#include "GGEMS/sources/GGEMSSource.hh"

class GGEMSParticles;
class GGEMSPseudoRandomGenerator;

/*!
  \class GGEMSSourceManager
  \brief GGEMS class handling the source(s)
*/
class GGEMS_EXPORT GGEMSSourceManager
{
  private:
    /*!
      \brief Unable the constructor for the user
    */
    GGEMSSourceManager(void);

    /*!
      \brief Unable the destructor for the user
    */
    ~GGEMSSourceManager(void);

  public:
    /*!
      \fn static GGEMSSourceManager& GetInstance(void)
      \brief Create at first time the Singleton
      \return Object of type GGEMSSourceManager
    */
    static GGEMSSourceManager& GetInstance(void)
    {
      static GGEMSSourceManager instance;
      return instance;
    }

  private:
    /*!
      \fn GGEMSSourceManager(GGEMSSourceManager const& source_manager) = delete
      \param source_manager - reference on the source manager
      \brief Avoid copy of the class by reference
    */
    GGEMSSourceManager(GGEMSSourceManager const& source_manager) = delete;

    /*!
      \fn GGEMSSourceManager& operator=(GGEMSSourceManager const& source_manager) = delete
      \param source_manager - reference on the source manager
      \brief Avoid assignement of the class by reference
    */
    GGEMSSourceManager& operator=(GGEMSSourceManager const& source_manager)
      = delete;

    /*!
      \fn GGEMSSourceManager(GGEMSSourceManager const&& source_manager) = delete
      \param source_manager - rvalue reference on the source manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSSourceManager(GGEMSSourceManager const&& source_manager) = delete;

    /*!
      \fn GGEMSSourceManager& operator=(GGEMSSourceManager const&& source_manager) = delete
      \param source_manager - rvalue reference on the source manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSSourceManager& operator=(GGEMSSourceManager const&& source_manager)
      = delete;

  public:
    /*!
      \fn void Store(GGEMSSource* p_source)
      \param p_source - pointer to GGEMS source
      \brief storing the source pointer to source manager
    */
    void Store(GGEMSSource* p_source);

    /*!
      \fn void Initialize(void) const
      \brief Initialize a GGEMS source
    */
    void Initialize(void) const;

    /*!
      \fn inline std::size_t GetNumberOfBatchs(void) const
      \return the number of batch of particle
      \brief method returning the number of particles by batch
    */
    inline std::size_t GetNumberOfBatchs(void) const
    {
      return p_sources_->GetNumberOfBatchs();
    }

    /*!
      \fn inline GGulong GetNumberOfParticlesInBatch(std::size_t const& batch_index)
      \return the number of particle for a specific batch
      \brief method returning the number of particles in a specific batch
    */
    inline GGulong GetNumberOfParticlesInBatch(std::size_t const& batch_index)
    {
      return p_sources_->GetNumberOfParticlesInBatch(batch_index);
    }

    /*!
      \fn GGEMSParticles* GetParticles(void) const
      \return pointer on particle stack
      \brief method returning the OpenCL stack on particles
    */
    inline GGEMSParticles* GetParticles(void) const {return p_particles_;}

    /*!
      \fn GGEMSPseudoRandomGenerator* GetPseudoRandomGenerator(void) const
      \return pointer on pseudo random stack
      \brief method returning the OpenCL stack on pseudo random numbers
    */
    inline GGEMSPseudoRandomGenerator* GetPseudoRandomGenerator(void) const
    {
      return p_pseudo_random_generator_;
    }

    /*!
      \fn void GetPrimaries(GGulong const& number_of_particles) const
      \param number_of_particles - number of particles to simulate
      \brief Generate primary particles for a specific source
    */
    inline void GetPrimaries(GGulong const& number_of_particles) const
    {
      p_sources_->GetPrimaries(number_of_particles);
    }

  private: // Source infos
    GGEMSSource* p_sources_; /*!< Pointer on GGEMS sources */
    GGuint number_of_sources_; /*!< Number of source */

  private: // Particle and random infos
    GGEMSParticles* p_particles_; /*!< Pointer on particle management */
    GGEMSPseudoRandomGenerator* p_pseudo_random_generator_; /*!< Pointer on pseudo random generator */
};

#endif // End of GUARD_GGEMS_SOURCES_GGEMSSOURCEMANAGER
