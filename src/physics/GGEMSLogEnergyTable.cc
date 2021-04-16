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
  \file GGEMSLogEnergyTable.cc

  \brief GGEMS class computing log table for cut convertion from length to energy.

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 24, 2020
*/

#include "GGEMS/physics/GGEMSLogEnergyTable.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/maths/GGEMSMathAlgorithms.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSLogEnergyTable::GGEMSLogEnergyTable(GGfloat const& lowest_energy, GGfloat const& highest_energy, GGsize const& number_of_bins)
{
  GGcout("GGEMSLogEnergyTable", "GGEMSLogEnergyTable", 3) << "GGEMSLogEnergyTable creating..." << GGendl;

  bin_width_ = logf(highest_energy / lowest_energy) / static_cast<GGfloat>(number_of_bins);
  base_bin_ = logf(lowest_energy) / bin_width_;

  number_of_nodes_ = number_of_bins + 1;

  loss_table_data_.reserve(number_of_nodes_);
  bins_.reserve(number_of_nodes_);

  // Begin of vector
  loss_table_data_.push_back(0.0f);
  bins_.push_back(lowest_energy);

  // Filling the vectors
  for (GGsize i = 1; i < number_of_nodes_-1; ++i) {
    loss_table_data_.push_back(0.0f);
    bins_.push_back(expf((base_bin_+static_cast<GGfloat>(i))*bin_width_));
  }

  // End of vector
  loss_table_data_.push_back(0.0f);
  bins_.push_back(highest_energy);

  // Storing edges
  edge_min_ = bins_.front();
  edge_max_ = bins_.back();

  GGcout("GGEMSLogEnergyTable", "GGEMSLogEnergyTable", 3) << "GGEMSLogEnergyTable created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSLogEnergyTable::~GGEMSLogEnergyTable(void)
{
  GGcout("GGEMSLogEnergyTable", "~GGEMSLogEnergyTable", 3) << "GGEMSLogEnergyTable erasing..." << GGendl;

  GGcout("GGEMSLogEnergyTable", "~GGEMSLogEnergyTable", 3) << "GGEMSLogEnergyTable erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSLogEnergyTable::SetValue(GGsize const& index, GGfloat const& value)
{
  loss_table_data_.at(index) = value;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat GGEMSLogEnergyTable::GetLossTableValue(GGfloat const& energy) const
{
  GGsize last_index = 0;
  GGfloat y = 0.0f;

  if (energy <= edge_min_) {
    last_index = 0;
    y = loss_table_data_.at(0);
  }
  else if (energy >= edge_max_) {
    last_index = number_of_nodes_ - 1;
    y = loss_table_data_.at(last_index);
  }
  else {
    last_index = FindBin(energy, last_index);
    y = LinearInterpolation(
      bins_.at(last_index), loss_table_data_.at(last_index),
      bins_.at(last_index+1), loss_table_data_.at(last_index+1),
      energy
    );
  }

  return y;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGsize GGEMSLogEnergyTable::FindBinLocation(GGfloat const& energy) const
{
  GGsize bin = static_cast<GGsize>(log(energy) / bin_width_ - base_bin_);

  if (bin + 2 > number_of_nodes_) {
    bin = number_of_nodes_ - 2;
  }
  else if (bin > 0 && energy < bins_.at(bin)) {
    --bin;
  }
  else if (bin + 2 < number_of_nodes_ && energy > bins_.at(bin+1)) {
    ++bin;
  }

  return bin;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGsize GGEMSLogEnergyTable::FindBin(GGfloat const& energy, GGsize const& index) const
{
  GGsize id = index;

  if (energy < bins_.at(1)) {
    id = 0;
  }
  else if (energy >= bins_.at(number_of_nodes_-2)) {
    id = number_of_nodes_ - 2;
  }
  else if (index >= number_of_nodes_ || energy < bins_.at(index) || energy > bins_.at(index+1)) {
    id = FindBinLocation(energy);
  }

  return id;
}
