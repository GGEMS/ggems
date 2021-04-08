#ifndef GUARD_GGEMS_SOURCES_GGEMSSOURCEMANAGER_HH
#define GUARD_GGEMS_SOURCES_GGEMSSOURCEMANAGER_HH

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

#include "GGEMS/physics/GGEMSParticles.hh"

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
      \fn GGsize GetNumberOfSources(void) const
      \brief Get the number of sources
      \return the number of sources
    */
    inline GGsize GetNumberOfSources(void) const {return number_of_sources_;}

    /*!
      \fn void Initialize(GGuint const& seed, bool const& is_tracking = false, GGint const& particle_tracking_id = 0) const
      \param seed - seed of the random
      \param is_tracking - boolean value for tracking
      \param particle_tracking_id - id of particle to track
      \brief Initialize a GGEMS source
    */
    void Initialize(GGuint const& seed, bool const& is_tracking = false, GGint const& particle_tracking_id = 0) const;

    /*!
      \fn inline std::string GetNameOfSource(GGsize const& source_index) const
      \param source_index - index of the source
      \return name of the source
      \brief get the name of the source
    */
    inline std::string GetNameOfSource(GGsize const& source_index) const {return sources_[source_index]->GetNameOfSource();}

    /*!
      \fn inline GGsize GetNumberOfBatchs(GGsize const& source_index, GGsize const& device_index) const
      \param source_index - index of the source
      \param device_index - index of activated device
      \return the number of batch of particle
      \brief method returning the number of particles by batch
    */
    inline GGsize GetNumberOfBatchs(GGsize const& source_index, GGsize const& device_index) const {return sources_[source_index]->GetNumberOfBatchs(device_index);}

    /*!
      \fn inline GGsize GetNumberOfParticlesInBatch(GGsize const& source_index, GGsize const& device_index, GGsize const& batch_index)
      \param source_index - index of the source
      \param device_index - index of activated device
      \param batch_index - index of the source
      \return the number of particle for a specific batch
      \brief method returning the number of particles in a specific batch
    */
    inline GGsize GetNumberOfParticlesInBatch(GGsize const& source_index, GGsize const& device_index, GGsize const& batch_index) {return sources_[source_index]->GetNumberOfParticlesInBatch(device_index, batch_index);}

    /*!
      \fn GGEMSParticles* GetParticles(void) const
      \return pointer on particle stack
      \brief method returning the OpenCL stack on particles
    */
    inline GGEMSParticles* GetParticles(void) const {return particles_;}

    /*!
      \fn GGEMSPseudoRandomGenerator* GetPseudoRandomGenerator(void) const
      \return pointer on pseudo random stack
      \brief method returning the OpenCL stack on pseudo random numbers
    */
    inline GGEMSPseudoRandomGenerator* GetPseudoRandomGenerator(void) const {return pseudo_random_generator_;}

    /*!
      \fn void GetPrimaries(GGsize const& source_index, GGsize const& thread_index, GGsize const& number_of_particles) const
      \param source_index - index of the source
      \param thread_index - index of activated device (thread index)
      \param number_of_particles - number of particles to simulate
      \brief Generate primary particles for a specific source
    */
    inline void GetPrimaries(GGsize const& source_index, GGsize const& thread_index, GGsize const& number_of_particles) const
    {
      particles_->SetNumberOfParticles(thread_index, number_of_particles);
      sources_[source_index]->GetPrimaries(thread_index, number_of_particles);
    }

    /*!
      \fn bool IsAlive(GGsize const& device_index) const
      \param device_index - index of activated device
      \return true if source is still alive, otherwize false
      \brief check if some particles are alive in OpenCL particle buffer
    */
    bool IsAlive(GGsize const& device_index) const;

    /*!
      \fn void Clean(void)
      \brief clean OpenCL data
    */
    void Clean(void);

  private: // Source infos
    GGEMSSource** sources_; /*!< Pointer on GGEMS sources */
    GGsize number_of_sources_; /*!< Number of sources */
    GGEMSParticles* particles_; /*!< Pointer on particle management */
    GGEMSPseudoRandomGenerator* pseudo_random_generator_; /*!< Pointer on pseudo random generator */
};

/*!
  \fn GGEMSSourceManager* get_instance_ggems_source_manager(void)
  \return the pointer on the singleton
  \brief Get the GGEMSSourceManager pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSSourceManager* get_instance_ggems_source_manager(void);

/*!
  \fn void initialize_source_manager(GGEMSSourceManager* source_manager, GGuint const& seed)
  \param source_manager - pointer on the singleton
  \param seed - seed of random
  \brief Initialize source
*/
extern "C" GGEMS_EXPORT void initialize_source_manager(GGEMSSourceManager* source_manager, GGuint const seed);

/*!
  \fn void print_infos_source_manager(GGEMSSourceManager* source_manager)
  \param source_manager - pointer on the singleton
  \brief Print information about source
*/
extern "C" GGEMS_EXPORT void print_infos_source_manager(GGEMSSourceManager* source_manager);

/*!
  \fn void clean_source_manager(GGEMSSourceManager* source_manager)
  \param source_manager - pointer on the singleton
  \brief Cleaning buffer
*/
extern "C" GGEMS_EXPORT void clean_source_manager(GGEMSSourceManager* source_manager);

#endif // End of GUARD_GGEMS_SOURCES_GGEMSSOURCEMANAGER
