#ifndef GUARD_GGEMS_PHYSICS_GGEMSMATERIALSMANAGER_HH
#define GUARD_GGEMS_PHYSICS_GGEMSMATERIALSMANAGER_HH

/*!
  \file GGEMSMaterialsManager.hh

  \brief GGEMS singleton class managing the material database

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Thrusday January 23, 2020
*/

#include "GGEMS/global/GGEMSExport.hh"

class GGEMSMaterialsDatabase;

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

  public:
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

  public:
    /*!
      \fn void SetMaterialsDatabase(char const* filename)
      \param filename - name of the file containing material database
      \brief set the material filename
    */
    void SetMaterialsDatabase(char const* filename);

  private:
    bool is_database_loaded_; /*!< Boolean checking if the database is loaded */
    GGEMSMaterialsDatabase* p_material_database_; /*!< Database of all materials */
};

/*!
  \fn GGEMSMaterialsManager* get_instance_materials_manager(void)
  \brief Get the GGEMSMaterialsManager pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSMaterialsManager*
  get_instance_materials_manager(void);

/*!
  \fn void set_process(GGEMSMaterialsManager* p_ggems_materials_manager, char const* filename)
  \param p_ggems_materials_manager - pointer on the singleton
  \param process_name - name of the process to activate
  \brief activate a specific process
*/
extern "C" GGEMS_EXPORT void set_materials_database_ggems_materials_manager(
  GGEMSMaterialsManager* p_ggems_materials_manager, char const* filename);

#endif // End of GUARD_GGEMS_PHYSICS_GGEMSMATERIALSMANAGER_HH
