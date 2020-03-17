#ifndef GUARD_GGEMS_MATERIALS_GGEMSMATERIALSDATABASEMANAGER_HH
#define GUARD_GGEMS_MATERIALS_GGEMSMATERIALSDATABASEMANAGER_HH

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

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSTypes.hh"
#include "GGEMS/tools/GGEMSTools.hh"
#include "GGEMS/global/GGEMSConstants.hh"

/*!
  \struct GGEMSChemicalElement
  \brief GGEMS structure managing a specific chemical element
*/
struct GGEMS_EXPORT GGEMSChemicalElement
{
  GGuchar atomic_number_Z_; /*!< Atomic number */
  GGfloat molar_mass_M_; /*!< Molar mass */
  GGfloat mean_excitation_energy_I_; /*!< Mean excitation energy */
  GGuchar state_; /*!< state of element GAS or SOLID */
  GGshort index_density_correction_; /*!< Index for density correction */
};

/*!
  \struct GGEMSSingleMaterial
  \brief GGEMS structure managing a specific material
*/
struct GGEMS_EXPORT GGEMSSingleMaterial
{
  std::vector<std::string> chemical_element_name_; /*! Name of the chemical elements */
  std::vector<GGfloat> mixture_f_; /*!< Fraction of element in material */
  GGfloat density_; /*!< Density of material */
  GGuchar nb_elements_; /*!< Number of elements in material */
};

typedef std::unordered_map<std::string, GGEMSChemicalElement> ChemicalElementUMap;
typedef std::unordered_map<std::string, GGEMSSingleMaterial> MaterialUMap;

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
      \fn GGEMSManager(GGEMSManager const& material_manager) = delete
      \param material_manager - reference on the material manager
      \brief Avoid copy of the class by reference
    */
    GGEMSMaterialsDatabaseManager(GGEMSMaterialsDatabaseManager const& material_manager) = delete;

    /*!
      \fn GGEMSManager& operator=(GGEMSManager const& material_manager) = delete
      \param material_manager - reference on the material manager
      \brief Avoid assignement of the class by reference
    */
    GGEMSMaterialsDatabaseManager& operator=(GGEMSMaterialsDatabaseManager const& material_manager) = delete;

    /*!
      \fn GGEMSManager(GGEMSManager const&& material_manager) = delete
      \param material_manager - rvalue reference on the material manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSMaterialsDatabaseManager(GGEMSMaterialsDatabaseManager const&& material_manager) = delete;

    /*!
      \fn GGEMSManager& operator=(GGEMSManager const&& material_manager) = delete
      \param material_manager - rvalue reference on the material manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSMaterialsDatabaseManager& operator=(GGEMSMaterialsDatabaseManager const&& material_manager) = delete;

    /*!
      \fn void SetMaterialsDatabase(char const* filename)
      \param filename - name of the file containing material database
      \brief set the material filename
    */
    void SetMaterialsDatabase(char const* filename);

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
      \fn bool IsReady(void) const
      \brief check if the material manager is ready
      \return true if the material manager is ready otherwize false
    */
    inline bool IsReady(void) const
    {
      if (materials_.empty()) {
        return false;
      }
      else {
        return true;
      }
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
      \param chemical_element - name of the chemical element
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
    void AddChemicalElements(std::string const& element_name, GGuchar const& element_Z, GGfloat const& element_M, GGfloat const& element_I, GGuchar const& state, GGshort const& index_density_correction);

  private:
    MaterialUMap materials_; /*!< Map storing the GGEMS materials */
    ChemicalElementUMap chemical_elements_; /*!< Map storing GGEMS chemical elements */
};

/*!
  \fn GGEMSMaterialsDatabaseManager* get_instance_materials_manager(void)
  \brief Get the GGEMSMaterialsDatabaseManager pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSMaterialsDatabaseManager* get_instance_materials_manager(void);

/*!
  \fn void set_materials_database_ggems_materials_manager(GGEMSMaterialsDatabaseManager* ggems_materials_manager, char const* filename)
  \param ggems_materials_manager - pointer on the singleton
  \param process_name - name of the process to activate
  \brief activate a specific process
*/
extern "C" GGEMS_EXPORT void set_materials_ggems_materials_manager(GGEMSMaterialsDatabaseManager* ggems_materials_manager, char const* filename);

/*!
  \fn void print_available_chemical_elements_ggems_materials_manager(GGEMSMaterialsDatabaseManager* ggems_materials_manager)
  \param p_ggems_materials_manager - pointer on the singleton
  \brief print all available chemical elements
*/
extern "C" GGEMS_EXPORT void print_available_chemical_elements_ggems_materials_manager(GGEMSMaterialsDatabaseManager* ggems_materials_manager);

/*!
  \fn void print_available_materials_ggems_materials_manager(GGEMSMaterialsDatabaseManager* ggems_materials_manager)
  \param p_ggems_materials_manager - pointer on the singleton
  \brief print all available materials
*/
extern "C" GGEMS_EXPORT void print_available_materials_ggems_materials_manager(GGEMSMaterialsDatabaseManager* ggems_materials_manager);

#endif // End of GUARD_GGEMS_PHYSICS_GGEMSMATERIALSDATABASEMANAGER_HH
