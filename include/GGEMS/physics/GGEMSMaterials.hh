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

#include <vector>
#include <unordered_map>

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSTypes.hh"

/*!
  \struct GGEMSSingleMaterial
  \brief GGEMS structure managing a specific material
*/
struct GGEMS_EXPORT GGEMSSingleMaterial
{
  /*!
    \brief GGEMSSingleMaterial constructor
  */
  GGEMSSingleMaterial(void);

  /*!
    \brief GGEMSSingleMaterial destructor
  */
  ~GGEMSSingleMaterial(void);

  /*!
    \fn GGEMSSingleMaterial(GGEMSSingleMaterial const& material) = delete
    \param material - reference on material
    \brief Avoid copy of the class by reference
  */
  GGEMSSingleMaterial(GGEMSSingleMaterial const& material) = delete;

  /*!
    \fn GGEMSSingleMaterial& operator=(GGEMSSingleMaterial const& material) = delete
    \param material - reference on material
    \brief Avoid assignement of the class by reference
  */
  GGEMSSingleMaterial& operator=(GGEMSSingleMaterial const& material) = delete;

  /*!
    \fn GGEMSSingleMaterial(GGEMSSingleMaterial const&& material) = delete
    \param material - reference on material
    \brief Avoid copy of the class by rvalue reference
  */
  GGEMSSingleMaterial(GGEMSSingleMaterial const&& material) = delete;

  /*!
    \fn GGEMSSingleMaterial& operator=(GGEMSSingleMaterial const&& material) = delete
    \param material - reference on material
    \brief Avoid copy of the class by rvalue reference
  */
  GGEMSSingleMaterial& operator=(GGEMSSingleMaterial const&& material) = delete;

  std::vector<std::string> mixture_Z_; /*! Atomic number (number of protons) by elements in material */
  std::vector<GGdouble> mixture_f_; /*!< Fraction of element in material */
  std::string name_; /*!< Name of material */
  GGdouble density_; /*!< Density of material */
  GGushort nb_elements_; /*!< Number of elements in material */
};

/*!
  \class GGEMSMaterialsDatabase
  \brief GGEMS class managing the complete material database
*/
class GGEMS_EXPORT GGEMSMaterialsDatabase
{
  public:
    /*!
      \brief GGEMSMaterialsDatabase constructor
    */
    GGEMSMaterialsDatabase(void);

    /*!
      \brief GGEMSMaterialsDatabase destructor
    */
    ~GGEMSMaterialsDatabase(void);

  public:
    /*!
      \fn GGEMSMaterialsDatabase(GGEMSMaterialsDatabase const& material_database) = delete
      \param material_database - reference on material database
      \brief Avoid copy of the class by reference
    */
    GGEMSMaterialsDatabase(GGEMSMaterialsDatabase const& material_database)
      = delete;

    /*!
      \fn GGEMSMaterialsDatabase& operator=(GGEMSMaterialsDatabase const& material_database) = delete
      \param material_database - reference on material database
      \brief Avoid assignement of the class by reference
    */
    GGEMSMaterialsDatabase& operator=(
      GGEMSMaterialsDatabase const& material_database) = delete;

    /*!
      \fn GGEMSMaterialsDatabase(GGEMSMaterialsDatabase const&& material_manager) = delete
      \param material_database - reference on material database
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSMaterialsDatabase(GGEMSMaterialsDatabase const&& material_database)
      = delete;

    /*!
      \fn GGEMSMaterialsDatabase& operator=(GGEMSMaterialsDatabase const&& material_database) = delete
      \param material_database - reference on material database
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSMaterialsDatabase& operator=(
      GGEMSMaterialsDatabase const&& material_database) = delete;

  public:
    /*!
      \fn void LoadMaterialsDatabase(std::string const& filename)
      \param filename - filename containing materials for GGEMS
      \brief Load materials for GGEMS
    */
    void LoadMaterialsDatabase(std::string const& filename);

  private:
};

#endif // End of GUARD_GGEMS_PHYSICS_GGEMSMATERIALS_HH
