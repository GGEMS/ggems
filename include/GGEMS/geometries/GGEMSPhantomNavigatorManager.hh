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

#include <memory>

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

    /*!
      \fn GGEMSPhantomNavigatorManager(GGEMSPhantomNavigatorManager const& phantom_navigator_manager) = delete
      \param phantom_navigator_manager - reference on the phantom navigator manager
      \brief Avoid copy of the class by reference
    */
    GGEMSPhantomNavigatorManager(GGEMSPhantomNavigatorManager const& phantom_navigator_manager) = delete;

    /*!
      \fn GGEMSPhantomNavigatorManager& operator=(GGEMSPhantomNavigatorManager const& source_manager) = delete
      \param phantom_navigator_manager - reference on the phantom navigator manager
      \brief Avoid assignement of the class by reference
    */
    GGEMSPhantomNavigatorManager& operator=(GGEMSPhantomNavigatorManager const& phantom_navigator_manager) = delete;

    /*!
      \fn GGEMSPhantomNavigatorManager(GGEMSPhantomNavigatorManager const&& source_manager) = delete
      \param phantom_navigator_manager - rvalue reference on the phantom navigator manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSPhantomNavigatorManager(GGEMSPhantomNavigatorManager const&& phantom_navigator_manager) = delete;

    /*!
      \fn GGEMSPhantomNavigatorManager& operator=(GGEMSPhantomNavigatorManager const&& source_manager) = delete
      \param phantom_navigator_manager - rvalue reference on the phantom navigator manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSPhantomNavigatorManager& operator=(GGEMSPhantomNavigatorManager const&& phantom_navigator_manager) = delete;

    /*!
      \fn void Store(std::shared_ptr<GGEMSPhantomNavigator> phantom_navigator)
      \param phantom_navigator - pointer to GGEMS phantom navigator
      \brief storing the phantom navigator pointer to phantom navigator manager
    */
    void Store(GGEMSPhantomNavigator* phantom_navigator);

    /*!
      \fn void Initialize(void) const
      \brief Initialize a GGEMS phantom
    */
    void Initialize(void) const;

    /*!
      \fn void PrintInfos(void)
      \brief Printing infos about the phantom navigators
    */
    void PrintInfos(void) const;

    /*!
      \fn std::size_t GetNumberOfPhantomNavigators(void) const
      \brief Get the number of phantom navigators
      \return the number of phantom navigators
    */
    inline std::size_t GetNumberOfPhantomNavigators(void) const {return phantom_navigators_.size();}

  private:
    std::vector<std::shared_ptr<GGEMSPhantomNavigator>> phantom_navigators_; /*!< Pointer on the phantom navigators */
};

/*!
  \fn GGEMSPhantomNavigatorManager* get_instance_ggems_phantom_navigator_manager(void)
  \brief Get the GGEMSPhantomNavigatorManager pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSPhantomNavigatorManager* get_instance_ggems_phantom_navigator_manager(void);

/*!
  \fn void print_infos_ggems_phantom_navigator_manager(GGEMSPhantomNavigatorManager* phantom_navigator_manager)
  \param phantom_navigator_manager - pointer on the phantom navigator manager
  \brief print infos about all declared phantom navigators
*/
extern "C" void GGEMS_EXPORT print_infos_ggems_phantom_navigator_manager(GGEMSPhantomNavigatorManager* phantom_navigator_manager);

#endif // End of GUARD_GGEMS_GEOMETRIES_GGEMSPHANTOMNAVIGATORMANAGER_HH
