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

#include <string>

#include "GGEMS/global/GGEMSOpenCLManager.hh"
#include "GGEMS/materials/GGEMSMaterialsDatabaseManager.hh"

#include "GGEMS/materials/GGEMSMaterialsStack.hh"

class GGEMSRangeCuts;

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
      \fn void AddMaterial(std::string const& material)
      \param material - name of the material
      \brief Add a material associated to a phantom
    */
    void AddMaterial(std::string const& material);

    /*!
      \fn inline std::string GetMaterialName(std::size_t i) const
      \param i - index of the material
      \return name of the material
      \brief get the name of the material at position i
    */
    inline std::string GetMaterialName(std::size_t i) const {return materials_.at(i);}

    /*!
      \fn inline std::size_t GetNumberOfMaterials(void) const
      \return the number of materials in the phantom
      \brief Get the number of materials in the phantom
    */
    inline std::size_t GetNumberOfMaterials(void) const {return materials_.size();}

    /*!
      \fn inline std::shared_ptr<cl::Buffer> GetMaterialTables(void) const
      \return the pointer on material tables on OpenCL device
      \brief get the pointer on material tables on OpenCL device
    */
    inline std::shared_ptr<cl::Buffer> GetMaterialTables(void) const {return material_tables_;}

    /*!
      \fn iinline std::shared_ptr<GGEMSRangeCuts> GetRangeCuts(void) const
      \brief get the pointer on range cuts
      \return the pointer on range cuts
    */
    inline std::shared_ptr<GGEMSRangeCuts> GetRangeCuts(void) const {return range_cuts_;}

    /*!
      \fn void SetLengthCut(std::string const& particle_name, GGfloat const& value, std::string const& unit)
      \param particle_name - type of particle gamma, e+, e-
      \param value - value of cut
      \param unit - length unit
      \brief set the cut for a particle in length
    */
    void SetLengthCut(std::string const& particle_name, GGfloat const& value, std::string const& unit);

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
    std::vector<std::string> materials_; /*!< Defined material for a phantom */
    std::shared_ptr<cl::Buffer> material_tables_; /*!< Material tables on OpenCL device */
    std::shared_ptr<GGEMSRangeCuts> range_cuts_; /*!< Cut for particles */

    // C++ singleton for OpenCL ang GGEMSMaterialsDatabase
    GGEMSOpenCLManager& opencl_manager_; /*!< Reference to OpenCL manager */
    GGEMSMaterialsDatabaseManager& material_manager_; /*!< Reference to material manager */
};

/*!
  \fn GGEMSMaterials* create_ggems_materials(void)
  \return the pointer on the singleton
  \brief Get the GGEMSMaterials pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSMaterials* create_ggems_materials(void);

/*!
  \fn void add_material_ggems_materials(GGEMSMaterials* materials, char const* material_name)
  \param materials - pointer on GGEMS materials
  \param material_name - name of the material
  \brief Add a material
*/
extern "C" GGEMS_EXPORT void add_material_ggems_materials(GGEMSMaterials* materials, char const* material_name);

/*!
  \fn void set_cut_ggems_materials(GGEMSMaterials* materials, char const* particle_type, GGfloat const cut, char const* unit)
  \param materials - pointer on GGEMS materials
  \param particle_type - type of particles : gamma, e+, e-
  \param cut - cut of the particle
  \param unit - length unit
  \brief Setting a cut
*/
extern "C" GGEMS_EXPORT void set_cut_ggems_materials(GGEMSMaterials* materials, char const* particle_type, GGfloat const cut, char const* unit);

/*!
  \fn void initialize_ggems_materials(GGEMSMaterials* materials)
  \param materials - pointer on GGEMS materials
  \brief Intialize the tables for the materials
*/
extern "C" GGEMS_EXPORT void initialize_ggems_materials(GGEMSMaterials* materials);

/*!
  \fn void print_material_properties_ggems_materials(GGEMSMaterials* materials)
  \param materials - pointer on GGEMS materials
  \brief Print tables
*/
extern "C" GGEMS_EXPORT void print_material_properties_ggems_materials(GGEMSMaterials* materials);

#endif // End of GUARD_GGEMS_PHYSICS_GGEMSMATERIALS_HH
