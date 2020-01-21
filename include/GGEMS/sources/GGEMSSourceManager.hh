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

class GGEMSSource;
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

  private: // Source infos
    GGEMSSource* p_sources_; /*!< Pointer on GGEMS sources */
    GGuint number_of_sources_; /*!< Number of source */

  private: // Particle and random infos
    GGEMSParticles* p_particle_; /*!< Pointer on particle management */
    GGEMSPseudoRandomGenerator* p_pseudo_random_generator_; /*!< Pointer on pseudo random generator */
};

#endif // End of GUARD_GGEMS_SOURCES_GGEMSSOURCEMANAGER
