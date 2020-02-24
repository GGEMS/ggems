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

#include <vector>
#include <unordered_map>

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSTypes.hh"

/*!
  \struct GGEMSChemicalElement
  \brief GGEMS structure managing a specific chemical element
*/
struct GGEMS_EXPORT GGEMSChemicalElement
{
  GGushort atomic_number_Z_; /*!< Atomic number */
  GGdouble atomic_mass_A_; /*!< Atomic mass */
  GGdouble mean_excitation_energy_I_; /*! Mean excitation energy */
};

/*!
  \struct GGEMSSingleMaterial
  \brief GGEMS structure managing a specific material
*/
struct GGEMS_EXPORT GGEMSSingleMaterial
{
  std::vector<std::string> mixture_Z_; /*! Atomic number (number of protons) by elements in material */
  std::vector<GGdouble> mixture_f_; /*!< Fraction of element in material */
  GGdouble density_; /*!< Density of material */
  GGushort nb_elements_; /*!< Number of elements in material */
};

typedef std::unordered_map<std::string, GGEMSChemicalElement> ChemicalElementUMap;
typedef std::unordered_map<std::string, GGEMSSingleMaterial> MaterialUMap;

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

    /*!
      \fn GGEMSManager(GGEMSManager const& material_manager) = delete
      \param material_manager - reference on the material manager
      \brief Avoid copy of the class by reference
    */
    GGEMSMaterialsManager(GGEMSMaterialsManager const& material_manager) = delete;

    /*!
      \fn GGEMSManager& operator=(GGEMSManager const& material_manager) = delete
      \param material_manager - reference on the material manager
      \brief Avoid assignement of the class by reference
    */
    GGEMSMaterialsManager& operator=(GGEMSMaterialsManager const& material_manager) = delete;

    /*!
      \fn GGEMSManager(GGEMSManager const&& material_manager) = delete
      \param material_manager - rvalue reference on the material manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSMaterialsManager(GGEMSMaterialsManager const&& material_manager) = delete;

    /*!
      \fn GGEMSManager& operator=(GGEMSManager const&& material_manager) = delete
      \param material_manager - rvalue reference on the material manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSMaterialsManager& operator=(GGEMSMaterialsManager const&& material_manager) = delete;

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
      \fn void AddChemicalElements(std::string const& element_name, GGushort const& element_Z, GGdouble const& element_A, GGdouble const& element_I_)
      \param element_name - Name of the element
      \param element_Z - Atomic number of the element
      \param element_A - Atomic mass of the element
      \param element_I - Mean excitation energy of the element
      \brief Adding a chemical element in GGEMS
    */
    void AddChemicalElements(std::string const& element_name, GGushort const& element_Z, GGdouble const& element_A, GGdouble const& element_I);

  private:
    MaterialUMap materials_; /*!< Map storing the GGEMS materials */
    ChemicalElementUMap chemical_elements_; /*!< Map storing GGEMS chemical elements */
};

/*!
  \fn GGEMSMaterialsManager* get_instance_materials_manager(void)
  \brief Get the GGEMSMaterialsManager pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSMaterialsManager* get_instance_materials_manager(void);

/*!
  \fn void set_materials_database_ggems_materials_manager(GGEMSMaterialsManager* ggems_materials_manager, char const* filename)
  \param ggems_materials_manager - pointer on the singleton
  \param process_name - name of the process to activate
  \brief activate a specific process
*/
extern "C" GGEMS_EXPORT void set_materials_ggems_materials_manager(GGEMSMaterialsManager* ggems_materials_manager, char const* filename);

/*!
  \fn void print_available_chemical_elements_ggems_materials_manager(GGEMSMaterialsManager* ggems_materials_manager)
  \param p_ggems_materials_manager - pointer on the singleton
  \brief print all available chemical elements
*/
extern "C" GGEMS_EXPORT void print_available_chemical_elements_ggems_materials_manager(GGEMSMaterialsManager* ggems_materials_manager);

/*!
  \fn void print_available_materials_ggems_materials_manager(GGEMSMaterialsManager* ggems_materials_manager)
  \param p_ggems_materials_manager - pointer on the singleton
  \brief print all available materials
*/
extern "C" GGEMS_EXPORT void print_available_materials_ggems_materials_manager(GGEMSMaterialsManager* ggems_materials_manager);

#endif // End of GUARD_GGEMS_PHYSICS_GGEMSMATERIALSMANAGER_HH
