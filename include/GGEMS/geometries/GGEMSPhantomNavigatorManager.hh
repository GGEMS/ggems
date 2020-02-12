#ifndef GUARD_GGEMS_GEOMETRIES_GGEMSPHANTOMNAVIGATORMANAGER_HH
#define GUARD_GGEMS_GEOMETRIES_GGEMSPHANTOMNAVIGATORMANAGER_HH

/*!
  \file GGEMSPhantomNavigatorManager.hh

  \brief GGEMS class handling the phantom navigators in GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday February 11, 2020
*/

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSTypes.hh"

class GGEMSPhantomNavigator;

/*!
  \class GGEMSPhantomNavigatorManager
  \brief GGEMS class handling the phantom navigator(s)
*/
class GGEMS_EXPORT GGEMSPhantomNavigatorManager
{
  private:
    /*!
      \brief Unable the constructor for the user
    */
    GGEMSPhantomNavigatorManager(void);

    /*!
      \brief Unable the destructor for the user
    */
    ~GGEMSPhantomNavigatorManager(void);

  public:
    /*!
      \fn static GGEMSPhantomNavigatorManager& GetInstance(void)
      \brief Create at first time the Singleton
      \return Object of type GGEMSPhantomNavigatorManager
    */
    static GGEMSPhantomNavigatorManager& GetInstance(void)
    {
      static GGEMSPhantomNavigatorManager instance;
      return instance;
    }

  private:
    /*!
      \fn GGEMSPhantomNavigatorManager(GGEMSPhantomNavigatorManager const& phantom_navigator_manager) = delete
      \param phantom_navigator_manager - reference on the phantom navigator manager
      \brief Avoid copy of the class by reference
    */
    GGEMSPhantomNavigatorManager(
      GGEMSPhantomNavigatorManager const& phantom_navigator_manager) = delete;

    /*!
      \fn GGEMSPhantomNavigatorManager& operator=(GGEMSPhantomNavigatorManager const& source_manager) = delete
      \param phantom_navigator_manager - reference on the phantom navigator manager
      \brief Avoid assignement of the class by reference
    */
    GGEMSPhantomNavigatorManager& operator=(
      GGEMSPhantomNavigatorManager const& phantom_navigator_manager) = delete;

    /*!
      \fn GGEMSPhantomNavigatorManager(GGEMSPhantomNavigatorManager const&& source_manager) = delete
      \param phantom_navigator_manager - rvalue reference on the phantom navigator manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSPhantomNavigatorManager(
      GGEMSPhantomNavigatorManager const&& phantom_navigator_manager) = delete;

    /*!
      \fn GGEMSPhantomNavigatorManager& operator=(GGEMSPhantomNavigatorManager const&& source_manager) = delete
      \param phantom_navigator_manager - rvalue reference on the phantom navigator manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSPhantomNavigatorManager& operator=(
      GGEMSPhantomNavigatorManager const&& phantom_navigator_manager) = delete;

  public:
    /*!
      \fn void Store(GGEMSPhantomNavigator* p_phantom_navigator)
      \param p_phantom_navigator - pointer to GGEMS phantom navigator
      \brief storing the phantom navigator pointer to phantom navigator manager
    */
    void Store(GGEMSPhantomNavigator* p_phantom_navigator);

  private:
    GGEMSPhantomNavigator** p_phantom_navigators_; /*!< Pointer on the phantom navigators */
    GGuint number_of_phantom_navigators_; /*!< Number of source */
};

#endif // End of GUARD_GGEMS_GEOMETRIES_GGEMSPHANTOMNAVIGATORMANAGER_HH
