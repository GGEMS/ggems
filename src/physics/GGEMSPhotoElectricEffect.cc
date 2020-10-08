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
#include "GGEMS/physics/GGEMSParticleCrossSectionsStack.hh"
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

GGfloat GGEMSPhotoElectricEffect::ComputeCrossSectionPerAtom(GGfloat const& energy, GGuchar const& atomic_number) const
{
  // Threshold at 10 eV
  GGdouble const kEmin = fmax(GGEMSSandiaTable::kIonizationPotentials[atomic_number], 10.0)*eV;
  if (energy < kEmin) return 0.0f;

  GGushort const kStart = GGEMSSandiaTable::kCumulativeIntervals[atomic_number-1];
  GGushort const kStop = kStart + GGEMSSandiaTable::kNumberOfIntervals[atomic_number];

  GGushort pos = kStop;
  while (energy < static_cast<GGfloat>(GGEMSSandiaTable::kSandiaTable[pos][0])*keV) --pos;

  GGdouble const kAoverAvo = ATOMIC_MASS_UNIT * static_cast<GGdouble>(atomic_number) / GGEMSSandiaTable::kZtoARatio[atomic_number];

  GGdouble const kREnergy = 1.0 / energy;
  GGdouble const kREnergy2 = kREnergy * kREnergy;

  return static_cast<GGfloat>(
    kREnergy * GGEMSSandiaTable::kSandiaTable[pos][1] * kAoverAvo * 0.160217648e-22 +
    kREnergy2 * GGEMSSandiaTable::kSandiaTable[pos][2] * kAoverAvo * 0.160217648e-25 +
    kREnergy * kREnergy2 * GGEMSSandiaTable::kSandiaTable[pos][3] * kAoverAvo * 0.160217648e-28 +
    kREnergy2 * kREnergy2 * GGEMSSandiaTable::kSandiaTable[pos][4] * kAoverAvo * 0.160217648e-31);
}
