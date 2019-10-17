#ifndef GUARD_GGEMS_SOURCES_GGEMSSOURCEDEFINITION_HH
#define GUARD_GGEMS_SOURCES_GGEMSSOURCEDEFINITION_HH

/*!
  \file ggems_source_definition.hh

  \brief GGEMS class managing the source in GGEMS, every new sources in GGEMS
  inherit from this class

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday October 15, 2019
*/

#include "GGEMS/global/ggems_export.hh"
#include "GGEMS/global/opencl_manager.hh"

/*!
  \class GGEMSSourceDefinition
  \brief GGEMS class managing the source in GGEMS, every new sources in GGEMS
  inherit from this class
*/
class GGEMS_EXPORT GGEMSSourceDefinition
{
  public:
    /*!
      \brief GGEMSSourceDefinition constructor
    */
    GGEMSSourceDefinition(void);

    /*!
      \brief GGEMSSourceDefinition destructor
    */
    virtual ~GGEMSSourceDefinition(void);

  public:
    /*!
      \fn GGEMSSourceDefinition(GGEMSSourceDefinition const& ggems_source) = delete
      \param ggems_manager - reference on the ggems manager
      \brief Avoid copy of the class by reference
    */
    GGEMSSourceDefinition(GGEMSSourceDefinition const& ggems_source) = delete;

    /*!
      \fn GGEMSSourceDefinition& operator=(GGEMSSourceDefinition const& ggems_source) = delete
      \param ggems_manager - reference on the ggems manager
      \brief Avoid assignement of the class by reference
    */
    GGEMSSourceDefinition& operator=(
      GGEMSSourceDefinition const& ggems_source) = delete;

    /*!
      \fn GGEMSSourceDefinition(GGEMSSourceDefinition const&& ggems_source) = delete
      \param ggems_manager - rvalue reference on the ggems manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSSourceDefinition(GGEMSSourceDefinition const&& ggems_source) = delete;

    /*!
      \fn GGEMSSourceDefinition& operator=(GGEMSSourceDefinition const&& ggems_source) = delete
      \param ggems_manager - rvalue reference on the ggems manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSSourceDefinition& operator=(
      GGEMSSourceDefinition const&& ggems_source) = delete;

  public:
    /*!
      \fn bool IsReady(void) const
      \return return false is the source is not ready
      \brief Check if the source is ready to be used
    */
    bool IsReady(void) const;

  protected: // Pure abstract method
    /*!
      \fn void GetPrimaries(cl::Buffer* p_primary_particles) = 0
      \param p_primary_particles - buffer of primary particles on OpenCL device
      \brief Generate primary particles
    */
    virtual void GetPrimaries(cl::Buffer* p_primary_particles) = 0;

    /*!
      \fn void Initialize(void) = 0
      \brief Initialize a GGEMS source
    */
    virtual void Initialize(void) = 0;

  protected:
    bool is_initialized_; /*!< Boolean checking if the source is initialized */
};

#endif // End of GUARD_GGEMS_SOURCES_GGEMSSOURCEDEFINITION_HH
