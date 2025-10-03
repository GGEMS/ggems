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
  \file GGEMSProgressBar.cc

  \brief GGEMS class displaying a progress bar into screen. This class is based on the progress.hpp header file of the C++ boost library.

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday June 23, 2021
*/

/// \cond
#include <iostream>
/// \endcond
///
#include "GGEMS/tools/GGEMSProgressBar.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSProgressBar::GGEMSProgressBar(GGsize const& expected_count)
: expected_count_(expected_count),
  count_(0),
  tic_(0),
  next_tic_count_(0)
{
  Restart(expected_count_);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSProgressBar::Restart(GGsize const& expected_count)
{
  // Initialize the private members
  expected_count_ = expected_count;
  count_ = 0;
  tic_ = 0;
  next_tic_count_ = 0;

  // Create the display bar
  std::cout << "0%   10   20   30   40   50   60   70   80   90   100%" << std::endl;
  std::cout << "|----|----|----|----|----|----|----|----|----|----|" << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSProgressBar& GGEMSProgressBar::operator++(void)
{
  this->operator+=(1);
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSProgressBar& GGEMSProgressBar::operator+=(GGsize const& increment)
{
  if ((count_ += increment) >= next_tic_count_) DisplayTic();
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSProgressBar::DisplayTic(void)
{
  GGsize const tics_needed = static_cast<GGsize>((static_cast<GGdouble>(count_) / static_cast<GGdouble>(expected_count_)) * 50.0);

  do {
    std::cout << "*" << std::flush;
  } while (++tic_ < tics_needed);

  next_tic_count_ = static_cast<GGsize>((static_cast<GGdouble>(tic_) / 50.0) * static_cast<GGdouble>(expected_count_));

  if (count_ == expected_count_) {
    if (tic_ < 51) std::cout << "*";
    std::cout << std::endl;
  }
}
