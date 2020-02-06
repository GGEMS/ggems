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
  \struct GGEMSChemicalElement
  \brief GGEMS structure managing a specific chemical element
*/
struct GGEMS_EXPORT GGEMSChemicalElement
{
  /*!
    \brief GGEMSChemicalElement constructor
  */
  GGEMSChemicalElement(void);

  /*!
    \brief GGEMSChemicalElement destructor
  */
  ~GGEMSChemicalElement(void);

  GGushort atomic_number_Z_; /*!< Atomic number */
  GGdouble atomic_mass_A_; /*!< Atomic mass */
  GGdouble mean_excitation_energy_I_; /*! Mean excitation energy */
};

typedef std::unordered_map<std::string, GGEMSChemicalElement>
  ChemicalElementUMap;

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

  std::vector<std::string> mixture_Z_; /*! Atomic number (number of protons) by elements in material */
  std::vector<GGdouble> mixture_f_; /*!< Fraction of element in material */
  std::string name_; /*!< Name of material */
  GGdouble density_; /*!< Density of material */
  GGushort nb_elements_; /*!< Number of elements in material */
};

typedef std::unordered_map<std::string, GGEMSSingleMaterial> MaterialUMap;

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

    /*!
      \fn void LoadChemicalElements(void)
      \brief load all the chemical elements
    */
    void LoadChemicalElements(void);

  private:
    /*!
      \fn void AddChemicalElements(std::string const& element_name, GGushort const& element_Z, GGdouble const& element_A, GGdouble const& element_I_)
      \param element_name - Name of the element
      \param element_Z - Atomic number of the element
      \param element_A - Atomic mass of the element
      \param element_I - Mean excitation energy of the element
      \brief Adding a chemical element in GGEMS
    */
    void AddChemicalElements(std::string const& element_name,
      GGushort const& element_Z, GGdouble const& element_A,
      GGdouble const& element_I);

  private:
    MaterialUMap materials_; /*!< Map storing the GGEMS materials */
    ChemicalElementUMap chemical_elements_; /*!< Map storing GGEMS chemical elements */
};

#endif // End of GUARD_GGEMS_PHYSICS_GGEMSMATERIALS_HH
