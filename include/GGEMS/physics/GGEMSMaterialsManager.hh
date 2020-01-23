#ifndef GUARD_GGEMS_PHYSICS_GGEMSMATERIALSMANAGER_HH
#define GUARD_GGEMS_PHYSICS_GGEMSMATERIALSMANAGER_HH

/*!
  \file GGEMSMaterialsManager.hh

  \brief GGEMS class managing the material database

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Thrusday January 23, 2020
*/

#include "GGEMS/global/GGEMSExport.hh"

/*!
  \class GGEMSMaterialsManager
  \brief GGEMS class managing the material database
*/
class GGEMS_EXPORT GGEMSMaterialsManager
{
  private:
    /*!
      \brief Unable the constructor for the user
    */
    GGEMSMaterialsManager(void);

    /*!
      \brief Unable the destructor for the user
    */
    ~GGEMSMaterialsManager(void);

  public:
    /*!
      \fn static GGEMSMaterialsManager& GetInstance(void)
      \brief Create at first time the Singleton
      \return Object of type GGEMSMaterialsManager
    */
    static GGEMSMaterialsManager& GetInstance(void)
    {
      static GGEMSMaterialsManager instance;
      return instance;
    }

  private:
    /*!
      \fn GGEMSManager(GGEMSManager const& material_manager) = delete
      \param material_manager - reference on the material manager
      \brief Avoid copy of the class by reference
    */
    GGEMSMaterialsManager(GGEMSMaterialsManager const& material_manager)
      = delete;

    /*!
      \fn GGEMSManager& operator=(GGEMSManager const& material_manager) = delete
      \param material_manager - reference on the material manager
      \brief Avoid assignement of the class by reference
    */
    GGEMSMaterialsManager& operator=(
      GGEMSMaterialsManager const& material_manager) = delete;

    /*!
      \fn GGEMSManager(GGEMSManager const&& material_manager) = delete
      \param material_manager - rvalue reference on the material manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSMaterialsManager(GGEMSMaterialsManager const&& material_manager)
      = delete;

    /*!
      \fn GGEMSManager& operator=(GGEMSManager const&& material_manager) = delete
      \param material_manager - rvalue reference on the material manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSMaterialsManager& operator=(
      GGEMSMaterialsManager const&& material_manager) = delete;
};

#endif // End of GUARD_GGEMS_PHYSICS_GGEMSMATERIALSMANAGER_HH
