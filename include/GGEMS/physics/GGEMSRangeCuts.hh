#ifndef GUARD_GGEMS_PHYSICS_GGEMSRANGECUTS_HH
#define GUARD_GGEMS_PHYSICS_GGEMSRANGECUTS_HH

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
  \file GGEMSRangeCuts.hh

  \brief GGEMS class storing and converting the cut in energy cut. The computations come from G4RToEConvForGamma, G4RToEConvForElectron, G4VRangeToEnergyConverter and G4PhysicsTable, G4PhysicsLogVector, G4PhysicsVector

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday March 18, 2020
*/

#ifdef _MSC_VER
#pragma warning(disable: 4251) // Deleting warning exporting STL members!!!
#endif

#include "GGEMS/materials/GGEMSMaterials.hh"

class GGEMSMaterials;
class GGEMSLogEnergyTable;

typedef std::unordered_map<std::string, GGfloat> EnergyCutUMap; /*!< Unordered map of material and energy cut */

/*!
  \class GGEMSRangeCuts
  \brief GGEMS class storing and converting the cut in energy cut
*/
class GGEMS_EXPORT GGEMSRangeCuts
{
  public:
    /*!
      \brief GGEMSRangeCuts constructor
    */
    GGEMSRangeCuts(void);

    /*!
      \brief GGEMSRangeCuts destructor
    */
    ~GGEMSRangeCuts(void);

    /*!
      \fn GGEMSRangeCuts(GGEMSRangeCuts const& range_cuts) = delete
      \param range_cuts - reference on the GGEMS range cuts
      \brief Avoid copy by reference
    */
    GGEMSRangeCuts(GGEMSRangeCuts const& range_cuts) = delete;

    /*!
      \fn GGEMSRangeCuts& operator=(GGEMSRangeCuts const& range_cuts) = delete
      \param range_cuts - reference on the GGEMS range cuts
      \brief Avoid assignement by reference
    */
    GGEMSRangeCuts& operator=(GGEMSRangeCuts const& range_cuts) = delete;

    /*!
      \fn GGEMSRangeCuts(GGEMSRangeCuts const&& range_cuts) = delete
      \param range_cuts - rvalue reference on the GGEMS range cuts
      \brief Avoid copy by rvalue reference
    */
    GGEMSRangeCuts(GGEMSRangeCuts const&& range_cuts) = delete;

    /*!
      \fn GGEMSRangeCuts& operator=(GGEMSRangeCuts const&& range_cuts) = delete
      \param range_cuts - rvalue reference on the GGEMS range cuts
      \brief Avoid copy by rvalue reference
    */
    GGEMSRangeCuts& operator=(GGEMSRangeCuts const&& range_cuts) = delete;

    /*!
      \fn void SetPhotonDistanceCut(GGfloat const& cut)
      \param cut - cut in length (mm)
      \brief set the photon length cut by the range cut manager
    */
    void SetPhotonDistanceCut(GGfloat const& cut);

    /*!
      \fn void SetElectronDistanceCut(GGfloat const& cut)
      \param cut - cut in length (mm)
      \brief set the electron length cut by the range cut manager
    */
    void SetElectronDistanceCut(GGfloat const& cut);

    /*!
      \fn void SetPositronDistanceCut(GGfloat const& cut)
      \param cut - cut in length (mm)
      \brief set the positron length cut by the range cut manager
    */
    void SetPositronDistanceCut(GGfloat const& cut);

    /*!
      \fn inline GGfloat GetPhotonDistanceCut(void) const
      \return the photon length cut in mm
      \brief get the photon length cut
    */
    inline GGfloat GetPhotonDistanceCut(void) const {return distance_cut_photon_;}

    /*!
      \fn inline EnergyCutUMap GetPhotonEnergyCut(void) const
      \return list of energy cut in each material
      \brief get the map of energy cut with material
    */
    inline EnergyCutUMap GetPhotonEnergyCut(void) const {return energy_cuts_photon_;}

    /*!
      \fn inline GGfloat GetElectronDistanceCut(void) const
      \return the electron length cut for photon
      \brief get the electron length cut for photon
    */
    inline GGfloat GetElectronDistanceCut(void) const {return distance_cut_electron_;}

    /*!
      \fn inline EnergyCutUMap GetElectronEnergyCut(void) const
      \return list of energy cut in each material for electron
      \brief get the map of energy cut with material for electron
    */
    inline EnergyCutUMap GetElectronEnergyCut(void) const {return energy_cuts_electron_;}

    /*!
      \fn inline GGfloat GetPositronDistanceCut(void) const
      \return the positron length cut in mm
      \brief get the positron length cut
    */
    inline GGfloat GetPositronDistanceCut(void) const {return distance_cut_positron_;}

    /*!
      \fn inline EnergyCutUMap GetPositronEnergyCut(void) const
      \return list of energy cut in each material for positron
      \brief get the map of energy cut with material for positron
    */
    inline EnergyCutUMap GetPositronEnergyCut(void) const {return energy_cuts_positron_;}

    /*!
      \fn void ConvertCutsFromDistanceToEnergy(GGEMSMaterials* materials)
      \param materials - pointer on the list of activated materials
      \brief Convert cut from length to energy
    */
    void ConvertCutsFromDistanceToEnergy(GGEMSMaterials* materials);

  private:
    /*!
      \fn GGfloat ConvertToEnergy(GGEMSMaterialTables* material_table, GGushort const& index_mat, std::string const& particle_name)
      \param material_table - material table on OpenCL device
      \param index_mat - index of the material
      \param particle_name - name of the particle
      \return energy cut of photon
      \brief Convert length cut to energy cut for gamma, e- and e+
    */
    GGfloat ConvertToEnergy(GGEMSMaterialTables* material_table, GGushort const& index_mat, std::string const& particle_name);

    /*!
      \fn void BuildElementsLossTable(GGEMSMaterialTables* material_table, GGushort const& index_mat, std::string const& particle_name)
      \param material_table - material table on OpenCL device
      \param index_mat - index of the material
      \param particle_name - name of the particle
      \brief Build loss table for elements in material
    */
    void BuildElementsLossTable(GGEMSMaterialTables* material_table, GGushort const& index_mat, std::string const& particle_name);

    /*!
      \fn void BuildAbsorptionLengthTable(GGEMSMaterialTables* material_table, GGushort const& index_mat)
      \param material_table - material table on OpenCL device
      \param index_mat - index of the material
      \brief Build absorption length table for photon
    */
    void BuildAbsorptionLengthTable(GGEMSMaterialTables* material_table, GGushort const& index_mat);

    /*!
      \fn void BuildMaterialLossTable(GGEMSMaterialTables* material_table, GGushort const& index_mat)
      \param material_table - material table on OpenCL device
      \param index_mat - index of the material
      \brief Build loss table for material in case of electron and positron
    */
    void BuildMaterialLossTable(GGEMSMaterialTables* material_table, GGushort const& index_mat);

    /*!
      \fn GGfloat ComputePhotonCrossSection(GGuchar const& atomic_number, GGfloat const& energy) const
      \param atomic_number - atomic number of the elements
      \param energy - energy of the bin
      \return cross secton value
      \brief compute cross secton value for photon depending on Z and energy
    */
    GGfloat ComputePhotonCrossSection(GGuchar const& atomic_number, GGfloat const& energy) const;

    /*!
      \fn GGfloat ComputeLossElectron(GGuchar const& atomic_number, GGfloat const& energy) const
      \param atomic_number - atomic number of the elements
      \param energy - energy of the bin
      \return loss energy, dE/dX
      \brief compute the loss de/dx for electron
    */
    GGfloat ComputeLossElectron(GGuchar const& atomic_number, GGfloat const& energy) const;

    /*!
      \fn GGfloat ComputeLossPositron(GGuchar const& atomic_number, GGfloat const& energy) const
      \param atomic_number - atomic number of the elements
      \param energy - energy of the bin
      \return loss energy, dE/dX
      \brief compute the loss de/dx for positron
    */
    GGfloat ComputeLossPositron(GGuchar const& atomic_number, GGfloat const& energy) const;

    /*!
      \fn GGfloat ConvertLengthToEnergyCut(GGfloat const& length_cut) const
      \param length_cut - length cut of the particle
      \return converted cut
      \brief convert length to energy cut
    */
    GGfloat ConvertLengthToEnergyCut(GGfloat const& length_cut) const;

  private:
    GGfloat min_energy_; /*!< Minimum energy of cross section table */
    GGfloat max_energy_; /*!< Maximum energy of cross section table */
    GGsize number_of_bins_; /*!< Number of bins in cross section table */

    // Photon
    GGfloat distance_cut_photon_; /*!< Photon cut in length */
    EnergyCutUMap energy_cuts_photon_; /*!< List of energy cuts for photon a material */

    // Electron
    GGfloat distance_cut_electron_; /*!< Electron cut in length */
    EnergyCutUMap energy_cuts_electron_; /*!< List of energy cuts for electron a material */

    // Positron
    GGfloat distance_cut_positron_; /*!< Positron cut in length */
    EnergyCutUMap energy_cuts_positron_; /*!< List of energy cuts for Positron a material */

    GGEMSLogEnergyTable* range_table_material_; /*!< Table of dE/dX for material */
    GGEMSLogEnergyTable** loss_table_dedx_table_elements_; /*!< Table of dE/dX for each element in materials */
    GGsize number_of_tables_; /*!< Number of tables in loss_table_dedx_table_elements_ */
};

#endif // GUARD_GGEMS_PHYSICS_GGEMSRANGECUTS_HH
