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
    GGEMSSourceManager& operator=(GGEMSSourceManager const& source_manager) = delete;

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
    GGEMSSourceManager& operator=(GGEMSSourceManager const&& source_manager) = delete;

    /*!
      \fn void Store(GGEMSSource* source)
      \param source - pointer to GGEMS source
      \brief storing the source pointer to source manager
    */
    void Store(GGEMSSource* source);

    /*!
      \fn void PrintInfos(void)
      \brief Printing infos about the sources
    */
    void PrintInfos(void) const;

    /*!
      \fn std::size_t GetNumberOfSources(void) const
      \brief Get the number of sources
      \return the number of sources
    */
    inline std::size_t GetNumberOfSources(void) const {return sources_.size();}

    /*!
      \fn void Initialize(void) const
      \brief Initialize a GGEMS source
    */
    void Initialize(void) const;

    /*!
      \fn inline std::size_t GetNumberOfBatchs(std::size_t const& source_index) const
      \param source_index - index of the source
      \return the number of batch of particle
      \brief method returning the number of particles by batch
    */
    inline std::size_t GetNumberOfBatchs(std::size_t const& source_index) const {return sources_[source_index]->GetNumberOfBatchs();}

    /*!
      \fn inline GGulong GetNumberOfParticlesInBatch(std::size_t const& source_index, std::size_t const& batch_index)
      \param source_index - index of the source
      \param batch_index - index of the source
      \return the number of particle for a specific batch
      \brief method returning the number of particles in a specific batch
    */
    inline GGulong GetNumberOfParticlesInBatch(std::size_t const& source_index, std::size_t const& batch_index) {return sources_[source_index]->GetNumberOfParticlesInBatch(batch_index);}

    /*!
      \fn GGEMSParticles* GetParticles(void) const
      \return pointer on particle stack
      \brief method returning the OpenCL stack on particles
    */
    inline GGEMSParticles* GetParticles(void) const {return particles_.get();}

    /*!
      \fn GGEMSPseudoRandomGenerator* GetPseudoRandomGenerator(void) const
      \return pointer on pseudo random stack
      \brief method returning the OpenCL stack on pseudo random numbers
    */
    inline GGEMSPseudoRandomGenerator* GetPseudoRandomGenerator(void) const {return pseudo_random_generator_.get();}

    /*!
      \fn void GetPrimaries(std::size_t const& source_index, GGulong const& number_of_particles) const
      \param source_index - index of the source
      \param number_of_particles - number of particles to simulate
      \brief Generate primary particles for a specific source
    */
    inline void GetPrimaries(std::size_t const& source_index, GGulong const& number_of_particles) const {sources_[source_index]->GetPrimaries(number_of_particles);}

    /*!
      \fn void AddSourceRAM(GGulong const& size)
      \param size - allocated RAM for sources in GGEMS
      \brief add RAM memory size for sources
    */
    void AddSourceRAM(GGulong const& size);

    /*!
      \fn void PrintAllocatedRAM(void) const
      \brief Print allocated RAM for sources
    */
    void PrintAllocatedRAM(void) const;

  private: // Source infos
    std::vector<std::shared_ptr<GGEMSSource>> sources_; /*!< Pointer on GGEMS sources */
    GGulong allocated_RAM_for_sources_; /*!< Allocated RAM in bytes for sources in GGEMS */

  private: // Particle and random infos
    std::shared_ptr<GGEMSParticles> particles_; /*!< Pointer on particle management */
    std::shared_ptr<GGEMSPseudoRandomGenerator> pseudo_random_generator_; /*!< Pointer on pseudo random generator */
};

/*!
  \fn GGEMSSourceManager* get_instance_ggems_source_manager(void)
  \return the pointer on the singleton
  \brief Get the GGEMSSourceManager pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSSourceManager* get_instance_ggems_source_manager(void);

/*!
  \fn void print_infos_ggems_source_manager(GGEMSSourceManager* source_manager)
  \param source_manager - pointer on the source manager
  \brief print infos about all declared sources
*/
extern "C" void GGEMS_EXPORT print_infos_ggems_source_manager(GGEMSSourceManager* source_manager);

#endif // End of GUARD_GGEMS_SOURCES_GGEMSSOURCEMANAGER
