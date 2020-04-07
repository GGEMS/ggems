/*!
  \file GGEMSLogEnergyTable.cc

  \brief GGEMS class computing log table for cut convertion from length to energy.

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 24, 2020
*/

#include <cmath>

#include "GGEMS/physics/GGEMSLogEnergyTable.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/maths/GGEMSMathAlgorithms.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSLogEnergyTable::GGEMSLogEnergyTable(GGfloat const& lowest_energy, GGfloat const& highest_energy, GGushort const& number_of_bins)
{
  GGcout("GGEMSLogEnergyTable", "GGEMSLogEnergyTable", 3) << "Allocation of GGEMSLogEnergyTable..." << GGendl;

  bin_width_ = logf(highest_energy / lowest_energy) / static_cast<GGfloat>(number_of_bins);
  base_bin_ = logf(lowest_energy) / bin_width_;

  number_of_nodes_ = number_of_bins + 1;

  loss_table_data_.reserve(number_of_nodes_);
  bins_.reserve(number_of_nodes_);

  // Begin of vector
  loss_table_data_.push_back(0.0f);
  bins_.push_back(lowest_energy);

  // Filling the vectors
  for (std::size_t i = 1; i < number_of_nodes_-1; ++i) {
    loss_table_data_.push_back(0.0f);
    bins_.push_back(expf((base_bin_+static_cast<GGfloat>(i))*bin_width_));
  }

  // End of vector
  loss_table_data_.push_back(0.0f);
  bins_.push_back(highest_energy);

  // Storing edges
  edge_min_ = bins_.front();
  edge_max_ = bins_.back();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSLogEnergyTable::~GGEMSLogEnergyTable(void)
{
  GGcout("GGEMSLogEnergyTable", "~GGEMSLogEnergyTable", 3) << "Deallocation of GGEMSLogEnergyTable..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSLogEnergyTable::SetValue(std::size_t const& index, GGfloat const& value)
{
  loss_table_data_.at(index) = value;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat GGEMSLogEnergyTable::GetLossTableValue(GGfloat const& energy) const
{
  std::size_t last_index = 0;
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

std::size_t GGEMSLogEnergyTable::FindBinLocation(GGfloat const& energy) const
{
  std::size_t bin = static_cast<std::size_t>(log(energy) / bin_width_ - base_bin_);

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

std::size_t GGEMSLogEnergyTable::FindBin(GGfloat const& energy, std::size_t const& index) const
{
  std::size_t id = index;

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
