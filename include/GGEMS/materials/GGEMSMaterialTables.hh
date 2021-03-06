#ifndef GUARD_GGEMS_MATERIALS_GGEMSMATERIALTABLES_HH
#define GUARD_GGEMS_MATERIALS_GGEMSMATERIALTABLES_HH

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
  \file GGEMSMaterialTables.hh

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
typedef struct GGEMSMaterialTables_t
{
  // Global parameters
  GGsize number_of_materials_; /*!< Number of the materials */
  GGsize total_number_of_chemical_elements_; /*!< Total number of chemical elements */

  // Infos by materials
  GGsize number_of_chemical_elements_[255]; /*!< Number of chemical elements in a single material */
  GGfloat density_of_material_[255]; /*!< Density of material in g/cm3 */
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
  GGfloat photon_energy_cut_[255]; /*!< Photon energy cut */
  GGfloat electron_energy_cut_[255]; /*!< Electron energy cut */
  GGfloat positron_energy_cut_[255]; /*!< Positron energy cut */

  // Infos by chemical elements by materials
  GGsize index_of_chemical_elements_[255]; /*!< Index to chemical element by material */
  GGuchar atomic_number_Z_[255*32]; /*!< Atomic number Z by chemical elements */
  GGfloat atomic_number_density_[255*32]; /*!< Atomic number density : fraction of element in material * density * Avogadro / Atomic mass */
  GGfloat mass_fraction_[255*32]; /*!< Mass fraction of element in material */
} GGEMSMaterialTables; /*!< Using C convention name of struct to C++ (_t deletion) */

#endif // GUARD_GGEMS_MATERIALS_GGEMSMATERIALSTABLE_HH
