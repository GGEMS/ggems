#ifndef GUARD_GGEMS_SOURCES_XRAYSOURCE_HH
#define GUARD_GGEMS_SOURCES_XRAYSOURCE_HH

/*!
  \file xray_source.hh

  \brief This class define a XRay source in GGEMS useful for CT/CBCT simulation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday October 22, 2019
*/

#include "GGEMS/global/ggems_export.hh"
#include "GGEMS/sources/ggems_source_manager.hh"
/*!
  \class XRaySource
  \brief This class define a XRay source in GGEMS useful for CT/CBCT simulation
*/
class GGEMS_EXPORT XRaySource : public GGEMSSourceManager
{
  public:
    /*!
      \fn static XRaySource* GetInstance()
      \brief Create at first time the Singleton
    */
    static XRaySource* GetInstance() {return new XRaySource;};

  protected:
    /*!
      \brief XRaySource constructor
    */
    XRaySource(void);

    /*!
      \brief XRaySource destructor
    */
    ~XRaySource(void);

  public:
    /*!
      \fn XRaySource(XRaySource const& xray_source) = delete
      \param ggems_manager - reference on the ggems manager
      \brief Avoid copy of the class by reference
    */
    XRaySource(XRaySource const& xray_source) = delete;

    /*!
      \fn XRaySource& operator=(XRaySource const& xray_source) = delete
      \param ggems_manager - reference on the ggems manager
      \brief Avoid assignement of the class by reference
    */
    XRaySource& operator=(XRaySource const& xray_source) = delete;

    /*!
      \fn XRaySource(XRaySource const&& xray_source) = delete
      \param ggems_manager - rvalue reference on the ggems manager
      \brief Avoid copy of the class by rvalue reference
    */
    XRaySource(XRaySource const&& xray_source) = delete;

    /*!
      \fn GGEMSSourceDefinition& operator=(GGEMSSourceDefinition const&& xray_source) = delete
      \param ggems_manager - rvalue reference on the ggems manager
      \brief Avoid copy of the class by rvalue reference
    */
    XRaySource& operator=(XRaySource const&& xray_source) = delete;

  public:
    /*!
      \fn void GetPrimaries(cl::Buffer* p_primary_particles)
      \param p_primary_particles - buffer of primary particles on OpenCL device
      \brief Generate primary particles
    */
    void GetPrimaries(cl::Buffer* p_primary_particles);

    /*!
      \fn void Initialize(void)
      \brief Initialize a GGEMS source
    */
    void Initialize(void);

  private:
    /*!
      \fn void CheckParameters(void) const
      \brief Check mandatory parameters for a X-Ray source
    */
    void CheckParameters(void) const;
};

/*!
  \fn XRaySource* create_ggems_xray_source(void)
  \brief Get the XRaySource pointer for python user.
*/
extern "C" GGEMS_EXPORT XRaySource* create_ggems_xray_source(void);

/*!
  \fn void delete_ggems_xray_source(void)
  \brief Delete the XRaySource pointer for python user
*/
extern "C" GGEMS_EXPORT void delete_ggems_xray_source(void);

/*!
  \fn void initialize_xray_source(XRaySource* source_manager)
  \param source_manager - pointer on the source
  \brief Initialize the X-Ray source
*/
extern "C" GGEMS_EXPORT void initialize_xray_source(
  XRaySource* p_source_manager);

#endif // End of GUARD_GGEMS_SOURCES_XRAYSOURCE_HH
