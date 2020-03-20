#ifndef GUARD_GGEMS_MATERIALS_GGEMSMATERIALS_HH
#define GUARD_GGEMS_MATERIALS_GGEMSMATERIALS_HH

/*!
  \file GGEMSMaterials.hh

  \brief GGEMS class handling material(s) for a specific navigator

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 4, 2020
*/

#ifdef _MSC_VER
#pragma warning(disable: 4251) // Deleting warning exporting STL members!!!
#endif

#include <set>
#include <string>

#include "GGEMS/global/GGEMSOpenCLManager.hh"
#include "GGEMS/materials/GGEMSMaterialsDatabaseManager.hh"

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSTypes.hh"
#include "GGEMS/materials/GGEMSMaterialsStack.hh"

/*!
  \class GGEMSMaterials
  \brief GGEMS class handling material(s) for a specific navigator
*/
class GGEMS_EXPORT GGEMSMaterials
{
  public:
    /*!
      \brief GGEMSMaterials constructor
    */
    GGEMSMaterials(void);

    /*!
      \brief GGEMSMaterials destructor
    */
    ~GGEMSMaterials(void);

    /*!
      \fn GGEMSMaterials(GGEMSMaterials const& materials) = delete
      \param materials - reference on the GGEMS materials
      \brief Avoid copy by reference
    */
    GGEMSMaterials(GGEMSMaterials const& materials) = delete;

    /*!
      \fn GGEMSMaterials& operator=(GGEMSMaterials const& materials) = delete
      \param materials - reference on the GGEMS materials
      \brief Avoid assignement by reference
    */
    GGEMSMaterials& operator=(GGEMSMaterials const& materials) = delete;

    /*!
      \fn GGEMSMaterials(GGEMSMaterials const&& materials) = delete
      \param materials - rvalue reference on the GGEMS materials
      \brief Avoid copy by rvalue reference
    */
    GGEMSMaterials(GGEMSMaterials const&& materials) = delete;

    /*!
      \fn GGEMSMaterials& operator=(GGEMSMaterials const&& materials) = delete
      \param materials - rvalue reference on the GGEMS materials
      \brief Avoid copy by rvalue reference
    */
    GGEMSMaterials& operator=(GGEMSMaterials const&& materials) = delete;

    /*!
      \fn bool AddMaterial(std::string const& material)
      \param material - name of the material
      \brief Add a material associated to a phantom
      \return false if material already added
    */
    bool AddMaterial(std::string const& material);

    /*!
      \fn void PrintInfos(void) const
      \brief printing labels and materials infos
    */
    void PrintInfos(void) const;

    /*!
      \fn void Initialize(void)
      \brief Initialize the materials for a navigator/phantom
    */
    void Initialize(void);

  private:
    /*!
      \fn void BuildMaterialTables(void)
      \brief Building material tables
    */
    void BuildMaterialTables(void);

  private:
    std::set<std::string> materials_; /*!< Defined material for a phantom */
    std::shared_ptr<cl::Buffer> material_tables_; /*!< Material tables on OpenCL device */

    // C++ singleton for OpenCL ang GGEMSMaterialsDatabase
    GGEMSOpenCLManager& opencl_manager_; /*!< Reference to OpenCL manager */
    GGEMSMaterialsDatabaseManager& material_manager_; /*!< Reference to material manager */
};

#endif // End of GUARD_GGEMS_PHYSICS_GGEMSMATERIALS_HH
