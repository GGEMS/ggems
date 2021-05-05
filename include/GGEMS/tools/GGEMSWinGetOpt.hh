#ifndef GUARD_GGEMS_TOOLS_GGEMSWINGETOPT_HH
#define GUARD_GGEMS_TOOLS_GGEMSWINGETOPT_HH

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
  \file GGEMSWinGetOpt.hh

  \brief Handle options for windows executable

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, Brest, FRANCE
  \version 1.0
  \date Wednesday May 5, 2021
*/

#ifdef _WIN32

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSTypes.hh"

/*
 * GNU-like getopt_long() and 4.4BSD getsubopt()/optreset extensions
 */

#define no_argument        0
#define required_argument  1
#define optional_argument  2

struct option
{
  const char *name;
  GGint has_arg;
  GGint *flag;
  GGint val;
};

GGint GGEMS_EXPORT getopt_long(GGint, char * const *, const char *, const struct option *, GGint *);
GGint GGEMS_EXPORT getopt_long_only(GGint, char * const *, const char *, const struct option *, GGint *);

#ifndef GUARD_GGEMS_TOOLS_GGEMSWINGETOPT_DEFINED
#define GUARD_GGEMS_TOOLS_GGEMSWINGETOPT_DEFINED

GGint GGEMS_EXPORT getopt(GGint, char * const *, const char *);
GGint GGEMS_EXPORT getsubopt(char **, char * const *, char **);
extern GGEMS_EXPORT char* optarg;
extern GGEMS_EXPORT GGint opterr;
extern GGEMS_EXPORT GGint optind;
extern GGEMS_EXPORT GGint optopt;
extern GGEMS_EXPORT GGint optreset;
extern GGEMS_EXPORT char* suboptarg;

#endif

#endif

#endif // GUARD_GGEMS_TOOLS_GGEMSWINGETOPT_HH

