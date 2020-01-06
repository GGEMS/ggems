
#ifndef GGEMS_EXPORT_H
#define GGEMS_EXPORT_H

#ifdef GGEMS_STATIC_DEFINE
#  define GGEMS_EXPORT
#  define GGEMS_NO_EXPORT
#else
#  ifndef GGEMS_EXPORT
#    ifdef ggems_EXPORTS
        /* We are building this library */
#      define GGEMS_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define GGEMS_EXPORT __declspec(dllimport)
#    endif
#  endif

#  ifndef GGEMS_NO_EXPORT
#    define GGEMS_NO_EXPORT 
#  endif
#endif

#ifndef GGEMS_DEPRECATED
#  define GGEMS_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef GGEMS_DEPRECATED_EXPORT
#  define GGEMS_DEPRECATED_EXPORT GGEMS_EXPORT GGEMS_DEPRECATED
#endif

#ifndef GGEMS_DEPRECATED_NO_EXPORT
#  define GGEMS_DEPRECATED_NO_EXPORT GGEMS_NO_EXPORT GGEMS_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef GGEMS_NO_DEPRECATED
#    define GGEMS_NO_DEPRECATED
#  endif
#endif

#endif /* GGEMS_EXPORT_H */
