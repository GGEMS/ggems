#ifndef GUARD_GGEMS_PHYSICS_GGEMSRANGECUTSMANAGER_HH
#define GUARD_GGEMS_PHYSICS_GGEMSRANGECUTSMANAGER_HH

/*!
  \file GGEMSRangeCutsManager.hh

  \brief GGEMS class managing the range cuts in GGEMS simulation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Friday March 6, 2020
*/

#include "GGEMS/global/GGEMSExport.hh"

/*!
  \class GGEMSRangeCutsManager
  \brief GGEMS class managing the range cuts in GGEMS simulation
*/
class GGEMS_EXPORT GGEMSRangeCutsManager
{
  private:
    /*!
      \brief Unable the constructor for the user
    */
    GGEMSRangeCutsManager(void);

    /*!
      \brief Unable the destructor for the user
    */
    ~GGEMSRangeCutsManager(void);

  public:
    /*!
      \fn static GGEMSRangeCutsManager& GetInstance(void)
      \brief Create at first time the Singleton
      \return Object of type GGEMSRangeCutsManager
    */
    static GGEMSRangeCutsManager& GetInstance(void)
    {
      static GGEMSRangeCutsManager instance;
      return instance;
    }

    /*!
      \fn GGEMSRangeCutsManager(GGEMSRangeCutsManager const& range_cuts_manager) = delete
      \param range_cuts_manager - reference on the range cuts manager
      \brief Avoid copy of the class by reference
    */
    GGEMSRangeCutsManager(GGEMSRangeCutsManager const& range_cuts_manager) = delete;

    /*!
      \fn GGEMSRangeCutsManager& operator=(GGEMSRangeCutsManager const& range_cuts_manager) = delete
      \param range_cuts_manager - reference on the ggems manager
      \brief Avoid assignement of the class by reference
    */
    GGEMSRangeCutsManager& operator=(GGEMSRangeCutsManager const& range_cuts_manager) = delete;

    /*!
      \fn GGEMSRangeCutsManager(GGEMSRangeCutsManager const&& range_cuts_manager) = delete
      \param range_cuts_manager - rvalue reference on the ggems manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSRangeCutsManager(GGEMSRangeCutsManager const&& range_cuts_manager) = delete;

    /*!
      \fn GGEMSRangeCutsManager& operator=(GGEMSRangeCutsManager const&& range_cuts_manager) = delete
      \param range_cuts_manager - rvalue reference on the ggems manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSRangeCutsManager& operator=(GGEMSRangeCutsManager const&& range_cuts_manager) = delete;

  private:
};

/*!
  \fn GGEMSRangeCutsManager* get_instance_range_cuts_manager(void)
  \brief Get the GGEMSRangeCutsManager pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSRangeCutsManager* get_instance_range_cuts_manager(void);

#endif // GUARD_GGEMS_PHYSICS_GGEMSRANGECUTSMANAGER_HH
