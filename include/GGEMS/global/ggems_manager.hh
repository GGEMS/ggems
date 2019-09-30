#ifndef GUARD_GGEMS_GLOBAL_GGEMS_HH
#define GUARD_GGEMS_GLOBAL_GGEMS_HH

/*!
  \file ggems.hh

  \brief GGEMS class managing the complete simulation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday September 30, 2019
*/

#include "GGEMS/global/ggems_export.hh"

class GGEMS_EXPORT GGEMS
{
  public:
    /*!
      \brief LookupTable constructor
    */
    GGEMS(void);

    /*!
      \brief LookupTable destructor
    */
    ~GGEMS(void);

  private:
    /*!
      \fn GGEMS(GGEMS const& ggems) = delete
      \param opencl_manager - reference on the singleton
      \brief Avoid copy of the singleton by reference
    */
    OpenCLManager(OpenCLManager const& opencl_manager) = delete;

    /*!
      \fn OpenCLManager& operator=(OpenCLManager const& opencl_manager) = delete
      \param opencl_manager - reference on the singleton
      \brief Avoid assignement of the singleton by reference
    */
    OpenCLManager& operator=(OpenCLManager const& opencl_manager) = delete;

    /*!
      \fn OpenCLManager(OpenCLManager const&& opencl_manager) = delete
      \param opencl_manager - rvalue reference on the singleton
      \brief Avoid copy of the singleton by rvalue reference
    */
    OpenCLManager(OpenCLManager const&& opencl_manager) = delete;

    /*!
      \fn OpenCLManager& operator=(OpenCLManager const&& opencl_manager) = delete
      \param opencl_manager - rvalue reference on the singleton
      \brief Avoid copy of the singleton by rvalue reference
    */
    OpenCLManager& operator=(OpenCLManager const&& opencl_manager) = delete;
};

#endif // End of GUARD_GGEMS_GLOBAL_GGEMS_HH