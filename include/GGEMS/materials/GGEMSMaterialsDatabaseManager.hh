#ifndef GUARD_GGEMS_MATERIALS_GGEMSMATERIALSDATABASEMANAGER_HH
#define GUARD_GGEMS_MATERIALS_GGEMSMATERIALSDATABASEMANAGER_HH

// ************************************************************************
// * This file is part of GGEMS.                                          *
// *                                                                      *
// * GGEMS is free software: you can redistribute it and/or modify        *
// * it under the terms of the GNU General Public License as published by *
// * the Free Software Foundation, either version 3 of the License, or    *
// * (at your option) any later version.                                  *
// *                                                                      *
// * GGEMS is distributed in the hope that it will be useful,             *
// * but WITHOUT ANY WARRANTY; without even the implied warranty of       *
// * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        *
// * GNU General Public License for more details.                         *
// *                                                                      *
// * You should have received a copy of the GNU General Public License    *
// * along with GGEMS.  If not, see <https://www.gnu.org/licenses/>.      *
// *                                                                      *
// ************************************************************************

/*!
  \file GGEMSMaterialsDatabaseManager.hh

  \brief GGEMS singleton class managing the material database

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Thrusday January 23, 2020
*/

#ifdef _MSC_VER
#pragma warning(disable: 4251) // Deleting warning exporting STL members!!!
#endif

#include <vector>
#include <unordered_map>
#include <sstream>

#include "GGEMS/global/GGEMSConstants.hh"

constant GGchar SOLID = 0; /*!< Solid state */
constant GGchar GAS = 1; /*!< Gas state */

/*!
  \struct GGEMSChemicalElement
  \brief GGEMS structure managing a specific chemical element
*/
struct GGEMS_EXPORT GGEMSChemicalElement
{
  GGchar atomic_number_Z_; /*!< Atomic number Z */
  GGfloat molar_mass_M_; /*!< Molar mass */
  GGfloat mean_excitation_energy_I_; /*!< Mean excitation energy */
  GGchar state_; /*!< state of element GAS or SOLID */
  GGchar index_density_correction_; /*!< Index for density correction */
};

/*!
  \struct GGEMSSingleMaterial
  \brief GGEMS structure managing a specific material
*/
struct GGEMS_EXPORT GGEMSSingleMaterial
{
  std::vector<std::string> chemical_element_name_; /*!< Name of the chemical elements */
  std::vector<GGfloat> mixture_f_; /*!< Fraction of element in material */
  GGfloat density_; /*!< Density of material */
  GGchar nb_elements_; /*!< Number of elements in material */
};

typedef std::unordered_map<std::string, GGEMSChemicalElement> ChemicalElementUMap; /*!< Unordered map with key : name of element, value the chemical element structure */
typedef std::unordered_map<std::string, GGEMSSingleMaterial> MaterialUMap; /*!< Unordered map with key : name of the material, value the material */

/*!
  \class GGEMSMaterialsDatabaseManager
  \brief GGEMS class managing the material database
*/
class GGEMS_EXPORT GGEMSMaterialsDatabaseManager
{
  private:
    /*!
      \brief Unable the constructor for the user
    */
    GGEMSMaterialsDatabaseManager(void);

    /*!
      \brief Unable the destructor for the user
    */
    ~GGEMSMaterialsDatabaseManager(void);

  public:
    /*!
      \fn static GGEMSMaterialsDatabaseManager& GetInstance(void)
      \brief Create at first time the Singleton
      \return Object of type GGEMSMaterialsDatabaseManager
    */
    static GGEMSMaterialsDatabaseManager& GetInstance(void)
    {
      static GGEMSMaterialsDatabaseManager instance;
      return instance;
    }

    /*!
      \fn GGEMSMaterialsDatabaseManager(GGEMSMaterialsDatabaseManager const& material_manager) = delete
      \param material_manager - reference on the material manager
      \brief Avoid copy of the class by reference
    */
    GGEMSMaterialsDatabaseManager(GGEMSMaterialsDatabaseManager const& material_manager) = delete;

    /*!
      \fn GGEMSMaterialsDatabaseManager& operator=(GGEMSMaterialsDatabaseManager const& material_manager) = delete
      \param material_manager - reference on the material manager
      \brief Avoid assignement of the class by reference
    */
    GGEMSMaterialsDatabaseManager& operator=(GGEMSMaterialsDatabaseManager const& material_manager) = delete;

    /*!
      \fn GGEMSMaterialsDatabaseManager(GGEMSMaterialsDatabaseManager const&& material_manager) = delete
      \param material_manager - rvalue reference on the material manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSMaterialsDatabaseManager(GGEMSMaterialsDatabaseManager const&& material_manager) = delete;

    /*!
      \fn GGEMSMaterialsDatabaseManager& operator=(GGEMSMaterialsDatabaseManager const&& material_manager) = delete
      \param material_manager - rvalue reference on the material manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSMaterialsDatabaseManager& operator=(GGEMSMaterialsDatabaseManager const&& material_manager) = delete;

    /*!
      \fn void SetMaterialsDatabase(std::string const* filename)
      \param filename - name of the file containing material database
      \brief set the material filename
    */
    void SetMaterialsDatabase(std::string const& filename);

    /*!
      \fn void PrintAvailableChemicalElements(void) const
      \brief Printing all the available elements
    */
    void PrintAvailableChemicalElements(void) const;

    /*!
      \fn void PrintAvailableMaterials(void) const
      \brief Printing all the available materials
    */
    void PrintAvailableMaterials(void) const;

    /*!
      \fn inline bool IsReady(void) const
      \brief check if the material manager is ready
      \return true if the material manager is ready otherwize false
    */
    inline bool IsReady(void) const
    {
      if (materials_.empty()) return false;
      else return true;
    }

    /*!
      \fn inline GGEMSSingleMaterial GetMaterial(std::string const& material_name) const
      \param material_name - name of the material
      \return the structure to a material
      \brief get the material
    */
    inline GGEMSSingleMaterial GetMaterial(std::string const& material_name) const
    {
      MaterialUMap::const_iterator iter = materials_.find(material_name);

      // Checking if the material exists
      if (iter == materials_.end())
      {
        std::ostringstream oss(std::ostringstream::out);
        oss << "Material '" << material_name << "' not found in the database!!!" << std::endl;
        GGEMSMisc::ThrowException("GGEMSMaterialsDatabaseManager", "GetMaterial", oss.str());
      }

      return iter->second;
    };

    /*!
      \fn inline GGEMSChemicalElement GetChemicalElement(std::string const& chemical_element_name) const
      \param chemical_element_name - name of the chemical element
      \return the structure to a chemical element
      \brief get the chemical element
    */
    inline GGEMSChemicalElement GetChemicalElement(std::string const& chemical_element_name) const
    {
      ChemicalElementUMap::const_iterator iter = chemical_elements_.find(chemical_element_name);

      // Checking if the material exists
      if (iter == chemical_elements_.end())
      {
        std::ostringstream oss(std::ostringstream::out);
        oss << "Chemical element '" << chemical_element_name << "' not found in the database!!!" << std::endl;
        GGEMSMisc::ThrowException("GGEMSMaterialsDatabaseManager", "GetChemicalElement", oss.str());
      }

      return iter->second;
    };

    /*!
      \fn GGfloat GetRadiationLength(std::string const& material) const
      \param material - name of the material
      \return the radiation length of a material
      \brief get the radiation length of a material
    */
    GGfloat GetRadiationLength(std::string const& material) const;

    /*!
      \fn inline GGfloat GetAtomicNumberDensity(std::string const& material, GGuchar const& index) const
      \param material - name of the material
      \param index - index of the element in the material
      \return get the atomic number density of an element in a material
      \brief Compute the atomic number density of an element in a material
    */
    inline GGfloat GetAtomicNumberDensity(std::string const& material, GGuchar const& index) const
    {
      // Getting the material
      GGEMSSingleMaterial const& single_material = GetMaterial(material);

      // Getting the specific chemical element
      GGEMSChemicalElement const& chemical_element = GetChemicalElement(single_material.chemical_element_name_[index]);

      // return the atomic number density, the number could be higher than float!!! Double is used
      return static_cast<GGfloat>(AVOGADRO / chemical_element.molar_mass_M_ * single_material.density_ * single_material.mixture_f_[index]);
    }

  private:
    /*!
      \fn void LoadMaterialsDatabase(std::string const& filename)
      \param filename - filename containing materials for GGEMS
      \brief Load materials for GGEMS
    */
    void LoadMaterialsDatabase(std::string const& filename);

    /*!
      \fn void LoadChemicalElements(void)
      \brief load all the chemical elements
    */
    void LoadChemicalElements(void);

    /*!
      \fn void AddChemicalElements(std::string const& element_name, GGuchar const& element_Z, GGfloat const& element_M, GGfloat const& element_I, GGuchar const& state, GGshort const& index_density_correction)
      \param element_name - Name of the element
      \param element_Z - Atomic number of the element
      \param element_M - Molar mass of the element
      \param element_I - Mean excitation energy of the element
      \param state - state of the material, gas/solid
      \param index_density_correction - index for the density correction
      \brief Adding a chemical element in GGEMS
    */
    void AddChemicalElements(std::string const& element_name, GGchar const& element_Z, GGfloat const& element_M, GGfloat const& element_I, GGchar const& state, GGchar const& index_density_correction);

  private:
    MaterialUMap materials_; /*!< Map storing the GGEMS materials */
    ChemicalElementUMap chemical_elements_; /*!< Map storing GGEMS chemical elements */
};

/*!
  \fn GGEMSMaterialsDatabaseManager* get_instance_materials_manager(void)
  \return the pointer on the singleton
  \brief Get the GGEMSMaterialsDatabaseManager pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSMaterialsDatabaseManager* get_instance_materials_manager(void);

/*!
  \fn void set_materials_ggems_materials_manager(GGEMSMaterialsDatabaseManager* ggems_materials_manager, char const* filename)
  \param ggems_materials_manager - pointer on the singleton
  \param filename - file with list of materials
  \brief enter the material database to GGEMS
*/
extern "C" GGEMS_EXPORT void set_materials_ggems_materials_manager(GGEMSMaterialsDatabaseManager* ggems_materials_manager, char const* filename);

/*!
  \fn void print_available_chemical_elements_ggems_materials_manager(GGEMSMaterialsDatabaseManager* ggems_materials_manager)
  \param ggems_materials_manager - pointer on the singleton
  \brief print all available chemical elements
*/
extern "C" GGEMS_EXPORT void print_available_chemical_elements_ggems_materials_manager(GGEMSMaterialsDatabaseManager* ggems_materials_manager);

/*!
  \fn void print_available_materials_ggems_materials_manager(GGEMSMaterialsDatabaseManager* ggems_materials_manager)
  \param ggems_materials_manager - pointer on the singleton
  \brief print all available materials
*/
extern "C" GGEMS_EXPORT void print_available_materials_ggems_materials_manager(GGEMSMaterialsDatabaseManager* ggems_materials_manager);

#endif // End of GUARD_GGEMS_PHYSICS_GGEMSMATERIALSDATABASEMANAGER_HH
