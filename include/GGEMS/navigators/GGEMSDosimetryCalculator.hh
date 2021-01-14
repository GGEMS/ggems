#ifndef GUARD_GGEMS_NAVIGATORS_GGEMSDOSIMETRYCALCULATOR_HH
#define GUARD_GGEMS_NAVIGATORS_GGEMSDOSIMETRYCALCULATOR_HH

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
  \file GGEMSDosimetryCalculator.hh

  \brief Class providing tools storing and computing dose in phantom

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \date Wednesday January 13, 2021
*/

#include "GGEMS/global/GGEMSExport.hh"

/*!
  \class GGEMSDosimetryCalculator
  \brief Class providing tools storing and computing dose in phantom
*/
class GGEMS_EXPORT GGEMSDosimetryCalculator
{
  public:
    /*!
      \brief GGEMSDosimetryCalculator constructor
    */
    GGEMSDosimetryCalculator(void);

    /*!
      \brief GGEMSDosimetryCalculator destructor
    */
    ~GGEMSDosimetryCalculator(void);

    /*!
      \fn GGEMSDosimetryCalculator(GGEMSDosimetryCalculator const& dose_calculator) = delete
      \param dose_calculator - reference on the GGEMS dose calculator
      \brief Avoid copy by reference
    */
    GGEMSDosimetryCalculator(GGEMSDosimetryCalculator const& dose_calculator) = delete;

    /*!
      \fn GGEMSDosimetryCalculator& operator=(GGEMSDosimetryCalculator const& dose_calculator) = delete
      \param dose_calculator - reference on the GGEMS dose calculator
      \brief Avoid assignement by reference
    */
    GGEMSDosimetryCalculator& operator=(GGEMSDosimetryCalculator const& dose_calculator) = delete;

    /*!
      \fn GGEMSDosimetryCalculator(GGEMSDosimetryCalculator const&& dose_calculator) = delete
      \param dose_calculator - rvalue reference on the GGEMS dose calculator
      \brief Avoid copy by rvalue reference
    */
    GGEMSDosimetryCalculator(GGEMSDosimetryCalculator const&& dose_calculator) = delete;

    /*!
      \fn GGEMSDosimetryCalculator& operator=(GGEMSDosimetryCalculator const&& dose_calculator) = delete
      \param dose_calculator - rvalue reference on the GGEMS dose calculator
      \brief Avoid copy by rvalue reference
    */
    GGEMSDosimetryCalculator& operator=(GGEMSDosimetryCalculator const&& dose_calculator) = delete;

    /*!
      \fn void Initialize(void)
      \brief Initialize dosimetry calculator class
    */
    void Initialize(void);

    private:
      /*!
        \fn void CheckParameters(void) const
        \return no returned value
      */
      void CheckParameters(void) const;
};

#endif // End of GUARD_GGEMS_NAVIGATORS_GGEMSDOSIMETRYCALCULATOR_HH
