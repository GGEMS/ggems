#ifndef GUARD_GGEMS_PHYSICS_GGEMSCOMPTONSCATTERING_HH
#define GUARD_GGEMS_PHYSICS_GGEMSCOMPTONSCATTERING_HH

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
  \file GGEMSComptonScattering.hh

  \brief Compton Scattering process from standard model for Geant4 (G4KleinNishinaCompton)

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 31, 2020
*/

#include "GGEMS/physics/GGEMSEMProcess.hh"

/*!
  \class GGEMSComptonScattering
  \brief Compton Scattering process from standard model for Geant4 (G4KleinNishinaCompton)
*/
class GGEMS_EXPORT GGEMSComptonScattering : public GGEMSEMProcess
{
  public:
    /*!
      \param primary_particle - type of primary particle (gamma)
      \param is_secondary - flag activating secondary (e- for Compton)
      \brief GGEMSComptonScattering constructor
    */
    GGEMSComptonScattering(std::string const& primary_particle = "gamma", bool const& is_secondary = false);

    /*!
      \brief GGEMSComptonScattering destructor
    */
    ~GGEMSComptonScattering(void) override;

    /*!
      \fn GGEMSComptonScattering(GGEMSComptonScattering const& compton_scattering) = delete
      \param compton_scattering - reference on the GGEMS compton scattering
      \brief Avoid copy by reference
    */
    GGEMSComptonScattering(GGEMSComptonScattering const& compton_scattering) = delete;

    /*!
      \fn GGEMSComptonScattering& operator=(GGEMSComptonScattering const& compton_scattering) = delete
      \param compton_scattering - reference on the GGEMS compton scattering
      \brief Avoid assignement by reference
    */
    GGEMSComptonScattering& operator=(GGEMSComptonScattering const& compton_scattering) = delete;

    /*!
      \fn GGEMSComptonScattering(GGEMSComptonScattering const&& compton_scattering) = delete
      \param compton_scattering - rvalue reference on the GGEMS compton scattering
      \brief Avoid copy by rvalue reference
    */
    GGEMSComptonScattering(GGEMSComptonScattering const&& compton_scattering) = delete;

    /*!
      \fn GGEMSComptonScattering& operator=(GGEMSComptonScattering const&& compton_scattering) = delete
      \param compton_scattering - rvalue reference on the GGEMS compton scattering
      \brief Avoid copy by rvalue reference
    */
    GGEMSComptonScattering& operator=(GGEMSComptonScattering const&& compton_scattering) = delete;

  private:
    /*!
      \fn GGfloat ComputeCrossSectionPerAtom(GGfloat const& energy, GGuchar const& atomic_number) const
      \param energy - energy of the bin
      \param atomic_number - Z number of the chemical element
      \return Compton cross section by atom
      \brief compute Compton cross section for an atom with Klein-Nishina
    */
    GGfloat ComputeCrossSectionPerAtom(GGfloat const& energy, GGuchar const& atomic_number) const override;
};

#endif // End of GUARD_GGEMS_PHYSICS_GGEMSCOMPTONSCATTERING_HH
