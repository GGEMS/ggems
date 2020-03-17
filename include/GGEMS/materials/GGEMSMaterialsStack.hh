#ifndef GUARD_GGEMS_PHYSICS_GGEMSMATERIALSSTACK_HH
#define GUARD_GGEMS_PHYSICS_GGEMSMATERIALSSTACK_HH

/*!
  \file GGEMSMaterialsStack.hh

  \brief Structure storing the material tables on OpenCL device

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday March 9, 2020
*/

#include "GGEMS/global/GGEMSConfiguration.hh"
#include "GGEMS/tools/GGEMSTypes.hh"

/*!
  \struct GGEMSMaterialTables_t
  \brief Structure storing the material tables on OpenCL device
*/
#ifdef OPENCL_COMPILER
typedef struct __attribute__((aligned (1))) GGEMSMaterialTables_t
#else
#ifdef _MSC_VER
#pragma pack(push, 1)
#endif
typedef struct PACKED GGEMSMaterialTables_t
#endif
{
  // Global parameters
  GGuchar number_of_materials_; /*!< Number of the materials */
  GGushort total_number_of_chemical_elements_; /*!< Total number of chemical elements */

  // Infos by materials
  GGuchar number_of_chemical_elements_[255]; /*!< Number of chemical elements in a single material */
  GGfloat density_of_material_[255]; /*! Density of material in g/cm3 */
  GGfloat number_of_atoms_by_volume_[255]; /*!< Number of atoms by volume */
  GGfloat number_of_electrons_by_volume_[255]; /*!< Number of electrons by volume */
  GGfloat mean_excitation_energy_[255]; /*!< Mean of excitation energy */
  GGfloat log_mean_excitation_energy_[255]; /*!< Log of mean of excitation energy */
  GGfloat radiation_length_[255]; /*!< Radiation length */
  GGfloat x0_density_[255]; /*!< x0 density correction */
  GGfloat x1_density_[255]; /*!< x1 density correction */
  GGfloat d0_density_[255]; /*!< d0 density correction */
  GGfloat c_density_[255]; /*!< c density correction */
  GGfloat a_density_[255]; /*!< a density correction */
  GGfloat m_density_[255]; /*!< m density correction */
  GGfloat f1_fluct_[255]; /*!< f1 energy loss fluctuation model */
  GGfloat f2_fluct_[255]; /*!< f2 energy loss fluctuation model */
  GGfloat energy0_fluct_[255]; /*!< energy 0 energy loss fluctuation model */
  GGfloat energy1_fluct_[255]; /*!< energy 1 energy loss fluctuation model */
  GGfloat energy2_fluct_[255]; /*!< energy 2 energy loss fluctuation model */
  GGfloat log_energy1_fluct_[255]; /*!< log of energy 0 energy loss fluctuation model */
  GGfloat log_energy2_fluct_[255]; /*!< log of energy 1 energy loss fluctuation model */
  GGfloat photon_energy_cut[255]; /*!< Photon energy cut */
  GGfloat electron_energy_cut[255]; /*!< Electron energy cut */

  // Infos by chemical elements by materials
  GGushort index_of_chemical_elements_[255]; /*!< Index to chemical element by material */
  GGuchar atomic_number_Z_[255*10]; /*!< Atomic number Z by chemical elements */
  GGfloat atomic_number_density_[255*10]; /*!< Atomic number density : fraction of element in material * density * Avogadro / Atomic mass */
  GGfloat mass_fraction_[255*10]; /*!< Mass fraction of element in material */
} GGEMSMaterialTables;
#ifndef OPENCL_COMPILER
#ifdef _MSC_VER
#pragma pack(pop)
#endif
#endif

#endif // GUARD_GGEMS_PHYSICS_GGEMSMATERIALSSTACK_HH
