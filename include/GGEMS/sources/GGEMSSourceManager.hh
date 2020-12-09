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
      \fn std::size_t GetNumberOfSources(void) const
      \brief Get the number of sources
      \return the number of sources
    */
    inline std::size_t GetNumberOfSources(void) const {return sources_.size();}

    /*!
      \fn void Initialize(GGuint const& seed) const
      \param seed - seed of the random
      \brief Initialize a GGEMS source
    */
    void Initialize(GGuint const& seed) const;

    /*!
      \fn inline std::string GetNameOfSource(std::size_t const& source_index) const
      \param source_index - index of the source
      \return name of the source
      \brief get the name of the source
    */
    inline std::string GetNameOfSource(std::size_t const& source_index) const {return sources_[source_index]->GetNameOfSource();}

    /*!
      \fn inline std::size_t GetNumberOfBatchs(std::size_t const& source_index) const
      \param source_index - index of the source
      \return the number of batch of particle
      \brief method returning the number of particles by batch
    */
    inline std::size_t GetNumberOfBatchs(std::size_t const& source_index) const {return sources_[source_index]->GetNumberOfBatchs();}

    /*!
      \fn inline GGlong GetNumberOfParticlesInBatch(std::size_t const& source_index, std::size_t const& batch_index)
      \param source_index - index of the source
      \param batch_index - index of the source
      \return the number of particle for a specific batch
      \brief method returning the number of particles in a specific batch
    */
    inline GGlong GetNumberOfParticlesInBatch(std::size_t const& source_index, std::size_t const& batch_index) {return sources_[source_index]->GetNumberOfParticlesInBatch(batch_index);}

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
    inline void GetPrimaries(std::size_t const& source_index, GGulong const& number_of_particles) const
    {
      particles_->SetNumberOfParticles(number_of_particles);
      sources_[source_index]->GetPrimaries(number_of_particles);
    }

    /*!
      \fn bool IsAlive(void) const
      \return true if source is still alive, otherwize false
      \brief check if some particles are alive in OpenCL particle buffer
    */
    bool IsAlive(void) const;

    /*!
      \fn void PrintKernelElapsedTime(void) const
      \brief Print elapsed time in kernel generating primaries for all the source
    */
    void PrintKernelElapsedTime(void) const;

  private: // Source infos
    std::vector<std::shared_ptr<GGEMSSource>> sources_; /*!< Pointer on GGEMS sources */
    std::shared_ptr<GGEMSParticles> particles_; /*!< Pointer on particle management */
    std::shared_ptr<GGEMSPseudoRandomGenerator> pseudo_random_generator_; /*!< Pointer on pseudo random generator */
};

#endif // End of GUARD_GGEMS_SOURCES_GGEMSSOURCEMANAGER
