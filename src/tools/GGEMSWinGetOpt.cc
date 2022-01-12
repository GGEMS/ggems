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
  \file GGEMSWinGetOpt.cc

  \brief Handle options for windows executable

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, Brest, FRANCE
  \version 1.0
  \date Wednesday May 5, 2021
*/

#ifdef _WIN32

#include <errno.h>
#include "GGEMS/tools/GGEMSWinGetOpt.hh"
#include <stdlib.h>
#include <string.h>

#define warnx(a, ...) (void)0

#define REPLACE_GETOPT /* use this getopt as the system getopt(3) */

#ifdef REPLACE_GETOPT
int opterr = 1; /* if error message should be printed */
int optind = 1; /* index into parent argv vector */
int optopt = '?'; /* character checked for validity */
int optreset; /* reset getopt */
char *optarg; /* argument associated with option */
#endif

#define PRINT_ERROR ((opterr) && (*options != ':'))

#define FLAG_PERMUTE 0x01 /* permute non-options to the end of argv */
#define FLAG_ALLARGS 0x02 /* treat non-options as args to option "-1" */
#define FLAG_LONGONLY 0x04 /* operate as getopt_long_only */

/* return values */
#define BADCH static_cast<int>('?')
#define BADARG ((*options == ':') ? static_cast<int>(':') : static_cast<int>('?'))
#define INORDER static_cast<int>(1)

#define EMSG ""

static int getopt_internal(int, char * const *, const char *, const struct option *, int *, int);
static int parse_long_options(char * const *, const char *, const struct option *, int *, int);
static int gcd(int, int);
static void permute_args(int, int, int, char * const *);

static char *place = const_cast<char*>(EMSG); /* option letter processing */

/* XXX: set optreset to 1 rather than these two */
static int nonopt_start = -1; /* first non option argument (for permute) */
static int nonopt_end = -1; /* first option after non options (for permute) */

/* Error messages */
// static const char recargchar[] = "option requires an argument -- %c";
// static const char recargstring[] = "option requires an argument -- %s";
// static const char ambig[] = "ambiguous option -- %.*s";
// static const char noarg[] = "option doesn't take an argument -- %.*s";
// static const char illoptchar[] = "unknown option -- %c";
// static const char illoptstring[] = "unknown option -- %s";

/*
 * Compute the greatest common divisor of a and b.
 */
static int
gcd(int a, int b)
{
  int c;

  c = a % b;
  while (c != 0) {
    a = b;
    b = c;
    c = a % b;
  }

  return (b);
}

/*
 * Exchange the block from nonopt_start to nonopt_end with the block
 * from nonopt_end to opt_end (keeping the same order of arguments
 * in each block).
 */
static void permute_args(int panonopt_start, int panonopt_end, int opt_end, char * const *nargv)
{
  int cstart, cyclelen, i, j, ncycle, nnonopts, nopts, pos;
  char *swap;

  // compute lengths of blocks and number and size of cycles
  nnonopts = panonopt_end - panonopt_start;
  nopts = opt_end - panonopt_end;
  ncycle = gcd(nnonopts, nopts);
  cyclelen = (opt_end - panonopt_start) / ncycle;

  for (i = 0; i < ncycle; i++) {
    cstart = panonopt_end+i;
    pos = cstart;
    for (j = 0; j < cyclelen; j++) {
      if (pos >= panonopt_end)
        pos -= nnonopts;
      else
        pos += nopts;
      swap = nargv[pos];
      /* LINTED const cast */
      (const_cast<char**>(nargv))[pos] = nargv[cstart];
      /* LINTED const cast */
      (const_cast<char**>(nargv))[cstart] = swap;
    }
  }
}

/*
 * parse_long_options --
 *	Parse long options in argc/argv argument vector.
 * Returns -1 if short_too is set and the option does not match long_options.
 */
static int parse_long_options(char * const *nargv, const char *options, const struct option *long_options, int *idx, int short_too)
{
  char *current_argv, *has_equal;
  size_t current_argv_len;
  int i, match;

  current_argv = place;
  match = -1;

  optind++;

  if ((has_equal = strchr(current_argv, '=')) != nullptr) {
    /* argument found (--option=arg) */
    current_argv_len = static_cast<size_t>(has_equal - current_argv);
    has_equal++;
  } else
    current_argv_len = strlen(current_argv);

  for (i = 0; long_options[i].name; i++) {
    /* find matching long option */
    if (strncmp(current_argv, long_options[i].name, current_argv_len))
      continue;

    if (strlen(long_options[i].name) == current_argv_len) {
      /* exact match */
      match = i;
      break;
    }
    /*
     * If this is a known short option, don't allow
     * a partial match of a single character.
     */
    if (short_too && current_argv_len == 1) continue;

    if (match == -1)	/* partial match */
      match = i;
    else {
      /* ambiguous abbreviation */
      if (PRINT_ERROR)
        warnx(ambig, (int)current_argv_len, current_argv);
      optopt = 0;
      return (BADCH);
    }
  }
  if (match != -1) {		/* option found */
    if (long_options[match].has_arg == no_argument && has_equal) {
      if (PRINT_ERROR)
        warnx(noarg, (int)current_argv_len, current_argv);
      // XXX: GNU sets optopt to val regardless of flag
      if (long_options[match].flag == nullptr)
        optopt = long_options[match].val;
      else
        optopt = 0;
      return (BADARG);
    }
    if (long_options[match].has_arg == required_argument || long_options[match].has_arg == optional_argument) {
      if (has_equal)
        optarg = has_equal;
      else if (long_options[match].has_arg == required_argument) {
        // optional argument doesn't use next nargv
        optarg = nargv[optind++];
      }
    }
    if ((long_options[match].has_arg == required_argument) && (optarg == nullptr)) {
      // Missing argument; leading ':' indicates no error should be generated.
      if (PRINT_ERROR)
        warnx(recargstring, current_argv);
      // XXX: GNU sets optopt to val regardless of flag
      if (long_options[match].flag == nullptr)
        optopt = long_options[match].val;
      else
        optopt = 0;
      --optind;
      return (BADARG);
    }
  } else { /* unknown option */
    if (short_too) {
      --optind;
      return (-1);
    }
    if (PRINT_ERROR)
      warnx(illoptstring, current_argv);
    optopt = 0;
    return (BADCH);
  }
  if (idx)
    *idx = match;
  if (long_options[match].flag) {
    *long_options[match].flag = long_options[match].val;
    return (0);
  } else
    return (long_options[match].val);
}

/*
 * getopt_internal --
 *	Parse argc/argv argument vector.  Called by user level routines.
 */
static int
getopt_internal(int nargc, char * const *nargv, const char *options, const struct option *long_options, int *idx, int flags)
{
  //char *oli;				/* option letter list index */
  int optchar = 0, short_too;
  static int posixly_correct = -1;

  if (options == nullptr) return (-1);

  /*
  * Disable GNU extensions if POSIXLY_CORRECT is set or options
  * string begins with a '+'.
  */

 if (posixly_correct == -1)
 {
    char* buf = nullptr;
    size_t sz = 0;
    posixly_correct = _dupenv_s( &buf, &sz, "POSIXLY_CORRECT" );
    //posixly_correct = (getenv("POSIXLY_CORRECT") != NULL);
    free(buf);
 }
 if (posixly_correct || *options == '+')
    flags &= ~FLAG_PERMUTE;
  else if (*options == '-')
    flags |= FLAG_ALLARGS;
  if (*options == '+' || *options == '-')
    options++;

  /*
   * XXX Some GNU programs (like cvs) set optind to 0 instead of
   * XXX using optreset.  Work around this braindamage.
  */
  if (optind == 0)
    optind = optreset = 1;

 optarg = nullptr;
 if (optreset)
    nonopt_start = nonopt_end = -1;
start:
 if (optreset || !*place) {		/* update scanning pointer */
    optreset = 0;
    if (optind >= nargc) {          /* end of argument vector */
      place = const_cast<char*>(EMSG);
      if (nonopt_end != -1) {
        /* do permutation, if we have to */
        permute_args(nonopt_start, nonopt_end, optind, nargv);
        optind -= nonopt_end - nonopt_start;
      }
      else if (nonopt_start != -1) {
        // If we skipped non-options, set optind to the first of them.
        optind = nonopt_start;
      }
      nonopt_start = nonopt_end = -1;
      return (-1);
    }
    if (*(place = nargv[optind]) != '-' || (place[1] == '\0' && strchr(options, '-') == nullptr)) {
      place = const_cast<char*>(EMSG);		/* found non-option */
      if (flags & FLAG_ALLARGS) {
        // GNU extension: return non-option as argument to option 1
        optarg = nargv[optind++];
        return (INORDER);
      }
      if (!(flags & FLAG_PERMUTE)) {
        // If no permutation wanted, stop parsing at first non-option.
        return (-1);
      }
      /* do permutation */
      if (nonopt_start == -1)
        nonopt_start = optind;
      else if (nonopt_end != -1) {
        permute_args(nonopt_start, nonopt_end, optind, nargv);
        nonopt_start = optind - (nonopt_end - nonopt_start);
        nonopt_end = -1;
      }
      optind++;
      /* process next argument */
      goto start;
    }
    if (nonopt_start != -1 && nonopt_end == -1)
      nonopt_end = optind;

    // If we have "-" do nothing, if "--" we are done.
    if (place[1] != '\0' && *++place == '-' && place[1] == '\0') {
      optind++;
      place = const_cast<char*>(EMSG);
      // We found an option (--), so if we skipped non-options, we have to permute.
      if (nonopt_end != -1) {
        permute_args(nonopt_start, nonopt_end, optind, nargv);
        optind -= nonopt_end - nonopt_start;
      }
      nonopt_start = nonopt_end = -1;
      return (-1);
    }
  }

  /*
   * Check long options if:
   *  1) we were passed some
   *  2) the arg is not just "-"
   *  3) either the arg starts with -- we are getopt_long_only()
   */
  if (long_options != nullptr && place != nargv[optind] && (*place == '-' || (flags & FLAG_LONGONLY))) {
    short_too = 0;
    if (*place == '-')
      place++;		/* --foo long option */
    else if (*place != ':' && strchr(options, *place) != nullptr)
      short_too = 1;		/* could be short option too */

    optchar = parse_long_options(nargv, options, long_options, idx, short_too);
    if (optchar != -1) {
      place = const_cast<char*>(EMSG);
      return (optchar);
    }
  }

 char const *oli = strchr(options, optchar);

  if ((optchar = static_cast<int>(*place++)) == static_cast<int>(':') || (optchar == static_cast<int>('-') && *place != '\0') || oli == nullptr) {
    /*
     * If the user specified "-" and  '-' isn't listed in
     * options, return -1 (non-option) as per POSIX.
     * Otherwise, it is an unknown option character (or ':').
     */
    if (optchar == static_cast<int>('-') && *place == '\0')
      return (-1);
    if (!*place)
      ++optind;
    if (PRINT_ERROR)
      warnx(illoptchar, optchar);
    optopt = optchar;
    return (BADCH);
  }
  if (long_options != nullptr && optchar == 'W' && oli[1] == ';') {
    /* -W long-option */
    if (*place) /* no space */
      /* NOTHING */;
    else if (++optind >= nargc) {	/* no arg */
      place = const_cast<char*>(EMSG);
      if (PRINT_ERROR)
        warnx(recargchar, optchar);
      optopt = optchar;
      return (BADARG);
    } else /* white space */
      place = nargv[optind];
    optchar = parse_long_options(nargv, options, long_options, idx, 0);
    place = const_cast<char*>(EMSG);
    return (optchar);
  }
  if (*++oli != ':') {			/* doesn't take argument */
    if (!*place)
      ++optind;
  } else { /* takes (optional) argument */
    optarg = nullptr;
    if (*place) /* no white space */
      optarg = place;
    else if (oli[1] != ':') {	/* arg not optional */
      if (++optind >= nargc) {	/* no arg */
        place = const_cast<char*>(EMSG);
        if (PRINT_ERROR) warnx(recargchar, optchar);
        optopt = optchar;
        return (BADARG);
     } else
        optarg = nargv[optind];
    }
    place = const_cast<char*>(EMSG);
    ++optind;
  }
  /* dump back option letter */
  return (optchar);
}

#ifdef REPLACE_GETOPT
/*
 * getopt --
 * Parse argc/argv argument vector.
 *
 * [eventually this will replace the BSD getopt]
 */
int getopt(int nargc, char * const *nargv, const char *options)
{
  /*
  * We don't pass FLAG_PERMUTE to getopt_internal() since
  * the BSD getopt(3) (unlike GNU) has never done this.
  *
  * Furthermore, since many privileged programs call getopt()
  * before dropping privileges it makes sense to keep things
  * as simple (and bug-free) as possible.
  */
  return (getopt_internal(nargc, nargv, options, nullptr, nullptr, 0));
}
#endif /* REPLACE_GETOPT */

/*
 * getopt_long --
 * Parse argc/argv argument vector.
 */
int getopt_long(int nargc, char * const *nargv, const char *options, const struct option *long_options, int *idx)
{
 return (getopt_internal(nargc, nargv, options, long_options, idx, FLAG_PERMUTE));
}

/*
 * getopt_long_only --
 * Parse argc/argv argument vector.
 */
int getopt_long_only(int nargc, char * const *nargv, const char *options, const struct option *long_options, int *idx)
{
  return (getopt_internal(nargc, nargv, options, long_options, idx, FLAG_PERMUTE|FLAG_LONGONLY));
}

#endif

