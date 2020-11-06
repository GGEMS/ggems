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
  \file GGEMSPhotoElectricEffect.cc

  \brief Photoelectric Effect process using Sandia table

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday April 13, 2020
*/

#include "GGEMS/materials/GGEMSMaterials.hh"
#include "GGEMS/physics/GGEMSPhotoElectricEffect.hh"
#include "GGEMS/physics/GGEMSParticleCrossSections.hh"
#include "GGEMS/physics/GGEMSSandiaTable.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSPhotoElectricEffect::GGEMSPhotoElectricEffect(std::string const& primary_particle, bool const& is_secondary)
: GGEMSEMProcess()
{
  GGcout("GGEMSPhotoElectricEffect", "GGEMSPhotoElectricEffect", 3) << "Allocation of GGEMSPhotoElectricEffect..." << GGendl;

  process_name_ = "Photoelectric";

  // Check type of primary particle
  if (primary_particle != "gamma") {
    std::ostringstream oss(std::ostringstream::out);
    oss << "For PhotoElectric effect, incident particle has to be a 'gamma'";
    GGEMSMisc::ThrowException("GGEMSPhotoElectricEffect", "GGEMSPhotoElectricEffect", oss.str());
  }

  process_id_ = PHOTOELECTRIC_EFFECT;
  primary_particle_ = "gamma";
  secondary_particle_ = "e-";
  is_secondaries_ = is_secondary;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSPhotoElectricEffect::~GGEMSPhotoElectricEffect(void)
{
  GGcout("GGEMSPhotoElectricEffect", "~GGEMSPhotoElectricEffect", 3) << "Deallocation of GGEMSPhotoElectricEffect..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat GGEMSPhotoElectricEffect::ComputeCrossSectionPerAtom(GGfloat const& energy, GGchar const& atomic_number) const
{
  // Threshold at 10 eV
  GGfloat e_min = fmax(GGEMSSandiaTable::kIonizationPotentials[atomic_number], 10.0f)*eV;
  if (energy < e_min) return 0.0f;

  GGshort start = GGEMSSandiaTable::kCumulativeIntervals[atomic_number-1];
  GGshort stop = start + GGEMSSandiaTable::kNumberOfIntervals[atomic_number];

  GGshort pos = stop;
  while (energy < static_cast<GGfloat>(GGEMSSandiaTable::kSandiaTable[pos][0])*keV) --pos;

  GGfloat aover_avo = ATOMIC_MASS_UNIT * static_cast<GGfloat>(atomic_number) / GGEMSSandiaTable::kZtoARatio[atomic_number];

  GGfloat energy_inv = 1.0f / energy;
  GGfloat energy_inv2 = energy_inv * energy_inv;

  return static_cast<GGfloat>(
    energy_inv * GGEMSSandiaTable::kSandiaTable[pos][1] * aover_avo * 0.160217648e-22 +
    energy_inv2 * GGEMSSandiaTable::kSandiaTable[pos][2] * aover_avo * 0.160217648e-25 +
    energy_inv * energy_inv2 * GGEMSSandiaTable::kSandiaTable[pos][3] * aover_avo * 0.160217648e-28 +
    energy_inv2 * energy_inv2 * GGEMSSandiaTable::kSandiaTable[pos][4] * aover_avo * 0.160217648e-31);
}
