#ifndef GUARD_GGEMS_PHYSICS_GGEMSMATERIALS_HH
#define GUARD_GGEMS_PHYSICS_GGEMSMATERIALS_HH

/*!
  \file GGEMSMaterial.hh

  \brief GGEMS class managing the complete material database and a specific
  material

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday February 3, 2020
*/

#include "GGEMS/global/GGEMSExport.hh"

/*!
  \class GGEMSSingleMaterial
  \brief GGEMS class managing a specific material
*/
class GGEMS_EXPORT GGEMSSingleMaterial
{
  public:
    /*!
      \brief GGEMSSingleMaterial constructor
    */
    GGEMSSingleMaterial(void);

    /*!
      \brief GGEMSSingleMaterial destructor
    */
    ~GGEMSSingleMaterial(void);

  private:
    /*!
      \fn GGEMSSingleMaterial(GGEMSSingleMaterial const& material) = delete
      \param material_database - reference on material database
      \brief Avoid copy of the class by reference
    */
    GGEMSSingleMaterial(GGEMSSingleMaterial const& material) = delete;

    /*!
      \fn GGEMSSingleMaterial& operator=(GGEMSSingleMaterial const& material) = delete
      \param material_database - reference on material database
      \brief Avoid assignement of the class by reference
    */
    GGEMSSingleMaterial& operator=(
      GGEMSSingleMaterial const& material) = delete;

    /*!
      \fn GGEMSSingleMaterial(GGEMSSingleMaterial const&& material_manager) = delete
      \param material_database - reference on material database
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSSingleMaterial(GGEMSSingleMaterial const&& material) = delete;

    /*!
      \fn GGEMSSingleMaterial& operator=(GGEMSSingleMaterial const&& material_database) = delete
      \param material_database - reference on material database
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSSingleMaterial& operator=(
      GGEMSSingleMaterial const&& material) = delete;

  private:
};

/*!
  \class GGEMSMaterialsDataBase
  \brief GGEMS class managing the complete material database
*/
class GGEMS_EXPORT GGEMSMaterialsDataBase
{
  public:
    /*!
      \brief GGEMSMaterialsDataBase constructor
    */
    GGEMSMaterialsDataBase(void);

    /*!
      \brief GGEMSMaterialsDataBase destructor
    */
    ~GGEMSMaterialsDataBase(void);

  private:
    /*!
      \fn GGEMSMaterialsDataBase(GGEMSMaterialsDataBase const& material_database) = delete
      \param material_database - reference on material database
      \brief Avoid copy of the class by reference
    */
    GGEMSMaterialsDataBase(GGEMSMaterialsDataBase const& material_database)
      = delete;

    /*!
      \fn GGEMSMaterialsDataBase& operator=(GGEMSMaterialsDataBase const& material_database) = delete
      \param material_database - reference on material database
      \brief Avoid assignement of the class by reference
    */
    GGEMSMaterialsDataBase& operator=(
      GGEMSMaterialsDataBase const& material_database) = delete;

    /*!
      \fn GGEMSMaterialsDataBase(GGEMSMaterialsDataBase const&& material_manager) = delete
      \param material_database - reference on material database
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSMaterialsDataBase(GGEMSMaterialsDataBase const&& material_database)
      = delete;

    /*!
      \fn GGEMSMaterialsDataBase& operator=(GGEMSMaterialsDataBase const&& material_database) = delete
      \param material_database - reference on material database
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSMaterialsDataBase& operator=(
      GGEMSMaterialsDataBase const&& material_database) = delete;

  private:
};

#endif // End of GUARD_GGEMS_PHYSICS_GGEMSMATERIALS_HH
