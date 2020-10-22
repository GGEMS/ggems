#ifndef GUARD_GGEMS_PHYSICS_GGEMSMATERIALSSTACK_HH
#define GUARD_GGEMS_PHYSICS_GGEMSMATERIALSSTACK_HH

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
#ifdef __OPENCL_C_VERSION__
typedef struct __attribute__((aligned (1))) GGEMSMaterialTables_t
#else
typedef struct PACKED GGEMSMaterialTables_t
#endif
{
  // Global parameters
  GGuchar number_of_materials_; /*!< Number of the materials */
  GGushort total_number_of_chemical_elements_; /*!< Total number of chemical elements */

  // Infos by materials
  GGuchar number_of_chemical_elements_[256]; /*!< Number of chemical elements in a single material */
  GGfloat density_of_material_[256]; /*!< Density of material in g/cm3 */
  GGfloat number_of_atoms_by_volume_[256]; /*!< Number of atoms by volume */
  GGfloat number_of_electrons_by_volume_[256]; /*!< Number of electrons by volume */
  GGfloat mean_excitation_energy_[256]; /*!< Mean of excitation energy */
  GGfloat log_mean_excitation_energy_[256]; /*!< Log of mean of excitation energy */
  GGfloat radiation_length_[256]; /*!< Radiation length */
  GGfloat x0_density_[256]; /*!< x0 density correction */
  GGfloat x1_density_[256]; /*!< x1 density correction */
  GGfloat d0_density_[256]; /*!< d0 density correction */
  GGfloat c_density_[256]; /*!< c density correction */
  GGfloat a_density_[256]; /*!< a density correction */
  GGfloat m_density_[256]; /*!< m density correction */
  GGfloat f1_fluct_[256]; /*!< f1 energy loss fluctuation model */
  GGfloat f2_fluct_[256]; /*!< f2 energy loss fluctuation model */
  GGfloat energy0_fluct_[256]; /*!< energy 0 energy loss fluctuation model */
  GGfloat energy1_fluct_[256]; /*!< energy 1 energy loss fluctuation model */
  GGfloat energy2_fluct_[256]; /*!< energy 2 energy loss fluctuation model */
  GGfloat log_energy1_fluct_[256]; /*!< log of energy 0 energy loss fluctuation model */
  GGfloat log_energy2_fluct_[256]; /*!< log of energy 1 energy loss fluctuation model */
  GGfloat photon_energy_cut_[256]; /*!< Photon energy cut */
  GGfloat electron_energy_cut_[256]; /*!< Electron energy cut */
  GGfloat positron_energy_cut_[256]; /*!< Positron energy cut */

  // Infos by chemical elements by materials
  GGushort index_of_chemical_elements_[256]; /*!< Index to chemical element by material */
  GGuchar atomic_number_Z_[256*16]; /*!< Atomic number Z by chemical elements */
  GGfloat atomic_number_density_[256*16]; /*!< Atomic number density : fraction of element in material * density * Avogadro / Atomic mass */
  GGfloat mass_fraction_[256*16]; /*!< Mass fraction of element in material */
} GGEMSMaterialTables; /*!< Using C convention name of struct to C++ (_t deletion) */

#endif // GUARD_GGEMS_PHYSICS_GGEMSMATERIALSSTACK_HH
