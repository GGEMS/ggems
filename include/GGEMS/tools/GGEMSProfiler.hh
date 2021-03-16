#ifndef GUARD_GGEMS_TOOLS_GGEMSPROFILER_HH
#define GUARD_GGEMS_TOOLS_GGEMSPROFILER_HH

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
  \file GGEMSProfiler.hh

  \brief GGEMS class handling a specific profiler type

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 16, 2021
*/

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSTypes.hh"

/*!
  \class GGEMSProfiler
  \brief GGEMS class handling a specific profiler type
*/
class GGEMS_EXPORT GGEMSProfiler
{
  public:
    /*!
      \brief GGEMSProfiler constructor
    */
    GGEMSProfiler(void);

    /*!
      \brief GGEMSProfiler destructor
    */
    ~GGEMSProfiler(void);

  private:
};

#endif // End of GUARD_GGEMS_TOOLS_GGEMSPROFILER_HH
