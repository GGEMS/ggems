#ifndef GUARD_GGEMS_TOOLS_GGEMSPROGRESSBAR_HH
#define GUARD_GGEMS_TOOLS_GGEMSPROGRESSBAR_HH

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
  \file GGEMSProgressBar.hh

  \brief GGEMS class displaying a progress bar into screen. This class is based on the progress.hpp header file of the C++ boost library.

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday June 23, 2021
*/

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSTypes.hh"

/*!
  \class GGEMSProgressBar
  \brief GGEMS class displaying a progress bar into screen
*/
class GGEMS_EXPORT GGEMSProgressBar
{
  public:
    /*!
      \param expected_count - Number of the expected step
      \brief GGEMSProgressBar constructor
    */
    explicit GGEMSProgressBar(GGsize const& expected_count);

    /*!
      \brief GGEMSProgressBar destructor
    */
    ~GGEMSProgressBar(void) {}

    /*!
      \fn GGEMSProgressBar(GGEMSProgressBar const& progress_bar) = delete
      \param progress_bar - reference on the GGEMS progress bar
      \brief Avoid copy by reference
    */
    GGEMSProgressBar(GGEMSProgressBar const& progress_bar) = delete;

    /*!
      \fn GGEMSProgressBar& operator=(GGEMSProgressBar const& progress_bar) = delete
      \param progress_bar - reference on the GGEMS progress bar
      \brief Avoid assignement by reference
    */
    GGEMSProgressBar& operator=(GGEMSProgressBar const& progress_bar) = delete;

    /*!
      \fn GGEMSProgressBar(GGEMSProgressBar const&& progress_bar) = delete
      \param progress_bar - rvalue reference on the GGEMS progress bar
      \brief Avoid copy by rvalue reference
    */
    GGEMSProgressBar(GGEMSProgressBar const&& progress_bar) = delete;

    /*!
      \fn GGEMSProgressBar& operator=(GGEMSProgressBar const&& progress_bar) = delete
      \param progress_bar - rvalue reference on the GGEMS progress bar
      \brief Avoid copy by rvalue reference
    */
    GGEMSProgressBar& operator=(GGEMSProgressBar const&& progress_bar) = delete;

    /*!
      \fn void Restart(GGsize const expected_count)
      \param expected_count - Number of the expected step
      \brief Restart the display
    */
    void Restart(GGsize const& expected_count);

    /*!
      \fn GGEMSProgressBar& operator++(void)
      \return reference to GGEMSProgressBar
      \brief Increment the display bar
    */
    GGEMSProgressBar& operator++(void);

  private:
    /*!
      \fn void DisplayTic(void)
      \brief Display the tics
    */
    void DisplayTic(void);

    /*!
      \fn GGEMSProgressBar& operator+=(GGsize const& increment)
      \param increment - counter for the tic
      \return reference to GGEMSProgressBar
      \brief Increment the display bar
    */
    GGEMSProgressBar& operator+=(GGsize const& increment);

  private:
    GGsize expected_count_; /*!< Expected number of the tics '*' */
    GGsize count_; /*!< Count of the tics '*' */
    GGsize tic_; /*!< Number of the tics */
    GGsize next_tic_count_; /*!< Next tic count */
};

#endif // End of GUARD_GGEMS_TOOLS_GGEMSPROGRESSBAR_HH
