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
  \file GGEMSPrint.cc

  \brief Print a custom std::cout end std::cerr handling verbosity

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, Brest, FRANCE
  \version 1.0
  \date Monday September 23, 2019
*/

#include "GGEMS/tools/GGEMSPrint.hh"

// Initializations
GGEMSStream GGcout = GGEMSStream(std::cout, GGEMSConsoleColor::green);
GGEMSStream GGcerr = GGEMSStream(std::cerr, GGEMSConsoleColor::red);
GGEMSStream GGwarn = GGEMSStream(std::cout, GGEMSConsoleColor::yellow);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSStream::GGEMSStream(std::ostream& stream, GGEMSConsoleColor const& color)
: class_name_(""),
  method_name_(""),
  verbosity_limit_(0),
  verbosity_level_(0),
  stream_counter_(0),
  stream_(stream),
  color_index_(color)
{}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSStream::SetVerbosity(GGint const& verbosity_limit)
{
  verbosity_limit_ = verbosity_limit;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSStream& GGEMSStream::operator()(std::string const& class_name,
  std::string const& method_name, GGint const& verbosity_level)
{
  class_name_ = class_name;
  method_name_ = method_name;
  verbosity_level_ = verbosity_level;
  stream_counter_ = 0;
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_ggems_verbose(GGint verbosity)
{
  GGcout.SetVerbosity(verbosity);
  GGcerr.SetVerbosity(verbosity);
  GGwarn.SetVerbosity(verbosity);
}
