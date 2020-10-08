#ifndef GUARD_GGEMS_PHYSICS_GGEMSLOGENERGYTABLE_HH
#define GUARD_GGEMS_PHYSICS_GGEMSLOGENERGYTABLE_HH

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
  \file GGEMSLogEnergyTable.hh

  \brief GGEMS class computing log table for cut convertion from length to energy.

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 24, 2020
*/

#ifdef _MSC_VER
#pragma warning(disable: 4251) // Deleting warning exporting STL members!!!
#endif

#include <vector>

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSTypes.hh"

/*!
  \class GGEMSLogEnergyTable
  \brief GGEMS class computing log table for cut convertion from length to energy.
*/
class GGEMS_EXPORT GGEMSLogEnergyTable
{
  public:
    /*!
      \param lowest_energy - lowest energy in the loss table
      \param highest_energy - highest energy in the loss table
      \param number_of_bins - number of bins in the loss table
      \brief GGEMSLogEnergyTable constructor
    */
    GGEMSLogEnergyTable(GGfloat const& lowest_energy, GGfloat const& highest_energy, GGushort const& number_of_bins);

    /*!
      \brief GGEMSLogEnergyTable destructor
    */
    ~GGEMSLogEnergyTable(void);

    /*!
      \fn GGEMSLogEnergyTable(GGEMSLogEnergyTable const& log_energy_table) = delete
      \param log_energy_table - reference on the GGEMS log energy table
      \brief Avoid copy by reference
    */
    GGEMSLogEnergyTable(GGEMSLogEnergyTable const& log_energy_table) = delete;

    /*!
      \fn GGEMSLogEnergyTable& operator=(GGEMSLogEnergyTable const& log_energy_table) = delete
      \param log_energy_table - reference on the GGEMS log energy table
      \brief Avoid assignement by reference
    */
    GGEMSLogEnergyTable& operator=(GGEMSLogEnergyTable const& log_energy_table) = delete;

    /*!
      \fn GGEMSLogEnergyTable(GGEMSLogEnergyTable const&& log_energy_table) = delete
      \param log_energy_table - rvalue reference on the GGEMS log energy table
      \brief Avoid copy by rvalue reference
    */
    GGEMSLogEnergyTable(GGEMSLogEnergyTable const&& log_energy_table) = delete;

    /*!
      \fn GGEMSLogEnergyTable& operator=(GGEMSLogEnergyTable const&& log_energy_table) = delete
      \param log_energy_table - rvalue reference on the GGEMS log energy table
      \brief Avoid copy by rvalue reference
    */
    GGEMSLogEnergyTable& operator=(GGEMSLogEnergyTable const&& log_energy_table) = delete;

    /*!
      \fn void SetValue(std::size_t const& index, GGfloat const& value)
      \param index - index of the bin
      \param value - value to put in data vector
      \brief Set the data vector at index
    */
    void SetValue(std::size_t const& index, GGfloat const& value);

    /*!
      \fn inline GGfloat GetEnergy(std::size_t const& index) const
      \param index - index of the bin
      \return energy at index
      \brief get the energy at the bin index
    */
    inline GGfloat GetEnergy(std::size_t const& index) const {return bins_.at(index);}

    /*!
      \fn inline GGfloat GetLossTableData(std::size_t const& index) const
      \param index - index of the bin
      \return loss table data value at index
      \brief get the loss table data value at the bin index
    */
    inline GGfloat GetLossTableData(std::size_t const& index) const {return loss_table_data_.at(index);}

    /*!
      \fn inline GGfloat GetLowEdgeEnergy(std::size_t const& index) const
      \param index - index of the bin
      \return energy at index
      \brief get the energy at the bin index similar to GGEMSLogEnergyTable::GetEnergy
    */
    inline GGfloat GetLowEdgeEnergy(std::size_t const& index) const {return bins_.at(index);}

    /*!
      \fn GGfloat GetLossTableValue(GGfloat const& energy) const
      \param energy - energy of the bin
      \return the interpolated loss table value
      \brief get the interpolated loss table value
    */
    GGfloat GetLossTableValue(GGfloat const& energy) const;

  private:
    /*!
      \fn std::size_t FindBinLocation(GGfloat const& energy) const
      \param energy - energy of the bin
      \return the bin where is the energy
      \brief get the bin where is the energy
    */
    std::size_t FindBinLocation(GGfloat const& energy) const;

    /*!
      \fn std::size_t FindBin(GGfloat const& energy, std::size_t const& index) const
      \param energy - energy of the bin
      \param index - index of the bin
      \return the bin where is the energy depending on a index
      \brief get the bin where is the energy depending on a index
    */
    std::size_t FindBin(GGfloat const& energy, std::size_t const& index) const;

  private:
    GGfloat edge_min_; /*!< Energy of first point */
    GGfloat edge_max_; /*!< Energy of the last point */
    std::size_t number_of_nodes_; /*!< Number of the nodes */
    std::vector<GGfloat> loss_table_data_; /*!< Vector keeping the crossection/energyloss */
    std::vector<GGfloat> bins_; /*!< Vector keeping energy */
    GGfloat bin_width_; /*!< Bin width - useful only for fixed binning */
    GGfloat base_bin_; /*!< Set this in constructor for performance */
};

#endif // GUARD_GGEMS_PHYSICS_GGEMSLOGENERGYTABLE_HH
