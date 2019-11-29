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

  public: // Pure abstract method from GGEMSSourceManager
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

    /*!
      \fn void PrintInfos(void)
      \brief Printing infos about the source
    */
    void PrintInfos(void) const;

  public:
    /*!
      \fn void SetBeamAperture(float const& beam_aperture)
      \param beam_aperture - beam aperture of the x-ray source
      \brief Set the beam aperture of the source
    */
    void SetBeamAperture(float const& beam_aperture);

    /*!
      \fn void SetFocalSpotSize(float const& width, float const& height, float const& depth)
      \param width - width of the focal spot size
      \param height - height of the focal spot size
      \param depth - depth of the focal spot size
      \brief Set the focal spot size of the x-ray source
    */
    void SetFocalSpotSize(float const& width, float const& height,
      float const& depth);

    /*!
      \fn void SetMonoenergy(float const& monoenergy)
      \param monoenergy - Monoenergy value
      \brief set the value of energy in monoenergy mode
    */
    void SetMonoenergy(float const& monoenergy);

    /*!
      \fn void SetPolyenergy(const char* energy_spectrum_filename)
      \param energy_spectrum_filename - filename containing the energy spectrum
      \brief set the energy spectrum file for polyenergy mode
    */
    void SetPolyenergy(const char* energy_spectrum_filename);

  private:
    /*!
      \fn void CheckParameters(void) const
      \brief Check mandatory parameters for a X-Ray source
    */
    void CheckParameters(void) const;

    /*!
      \fn void FillEnergy(void)
      \brief fill energy for poly or mono energy mode
    */
    void FillEnergy(void);

    /*!
      \fn void InitializeKernel(void)
      \brief Initialize kernel for XRaySource in OpenCL
    */
    void InitializeKernel(void);

  private: // Specific members for XRaySource
    float beam_aperture_; /*!< Beam aperture of the x-ray source */
    cl_float3 focal_spot_size_; /*!< Focal spot size of the x-ray source */
    bool is_monoenergy_mode_; /*!< Boolean checking the mode of energy */
    float monoenergy_; /*!< Monoenergy mode */
    std::string energy_spectrum_filename_; /*!< The energy spectrum filename for polyenergetic mode */
    cl_uint number_of_energy_bins_; /*!< Number of energy bins for the polyenergetic mode */
    cl::Buffer* p_energy_spectrum_; /*!< Energy spectrum for OpenCL device */
    cl::Buffer* p_cdf_; /*!< Cumulative distribution function to generate a random energy */
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

/*!
  \fn void set_position_xray_source(XRaySource* p_source_manager, float const pos_x, float const pos_y, float const pos_z)
  \param source_manager - pointer on the source
  \param pos_x - Position of the source in X
  \param pos_y - Position of the source in Y
  \param pos_z - Position of the source in Z
  \brief Set the position of the source in the global coordinates
*/
extern "C" GGEMS_EXPORT void set_position_xray_source(
  XRaySource* p_source_manager, float const pos_x, float const pos_y,
  float const pos_z);

/*!
  \fn void print_infos_xray_source(XRaySource const* p_source_manager)
  \param p_source_manager - pointer on the source
  \brief Print infos about the XRaySource
*/
extern "C" GGEMS_EXPORT void print_infos_xray_source(
  XRaySource* p_source_manager);

/*!
  \fn void SetParticleType_xray_source(XRaySource* p_source_manager, char const* particle_name)
  \param source_manager - pointer on the source
  \param particle_name - name/type of the particle: photon or electron
  \brief Set the type of the particle
*/
extern "C" GGEMS_EXPORT void set_particle_type_xray_source(
  XRaySource* p_source_manager, char const* particle_name);

/*!
  \fn void set_beam_aperture_xray_source(XRaySource* p_source_manager, float const beam_aperture)
  \param p_source_manager - pointer on the source
  \param beam_aperture - beam aperture of the x-ray source
  \brief set the beam aperture of the x-ray source
*/
extern "C" GGEMS_EXPORT void set_beam_aperture_xray_source(
  XRaySource* p_source_manager, float const beam_aperture);

/*!
  \fn void set_focal_spot_size_xray_source(XRaySource* p_source_manager, float const width, float const height, float const depth)
  \param p_source_manager - pointer on the source
  \param width - width of the focal spot size
  \param height - height of the focal spot size
  \param depth - depth of the focal spot size
  \brief Set the focal spot size of the x-ray source
*/
extern "C" GGEMS_EXPORT void set_focal_spot_size_xray_source(
  XRaySource* p_source_manager, float const width, float const height,
  float const depth);

/*!
  \fn void set_local_axis_xray_source(XRaySource* p_source_manager, float const m00, float const m01, float const m02, float const m10, float const m11, float const m12, float const m20, float const m21, float const m22)
  \param p_source_manager - pointer on the source
  \param m00 - Element 0,0 in the matrix 3x3 for local axis
  \param m01 - Element 0,1 in the matrix 3x3 for local axis
  \param m02 - Element 0,2 in the matrix 3x3 for local axis
  \param m10 - Element 1,0 in the matrix 3x3 for local axis
  \param m11 - Element 1,1 in the matrix 3x3 for local axis
  \param m12 - Element 1,2 in the matrix 3x3 for local axis
  \param m20 - Element 2,0 in the matrix 3x3 for local axis
  \param m21 - Element 2,1 in the matrix 3x3 for local axis
  \param m22 - Element 2,2 in the matrix 3x3 for local axis
  \brief Set the local axis element describing the source compared to global axis (center of world)
*/
extern "C" GGEMS_EXPORT void set_local_axis_xray_source(
  XRaySource* p_source_manager,
  float const m00, float const m01, float const m02,
  float const m10, float const m11, float const m12,
  float const m20, float const m21, float const m22);

/*!
  \fn void set_rotation_xray_source(XRaySource* p_source_manager, float const rx, float const ry, float const rz)
  \param p_source_manager - pointer on the source
  \param rx - Rotation around X along global axis
  \param ry - Rotation around Y along global axis
  \param rz - Rotation around Z along global axis
  \brief Set the rotation of the source around global axis
*/
extern "C" GGEMS_EXPORT void set_rotation_xray_source(
  XRaySource* p_source_manager, float const rx, float const ry,
  float const rz);

/*!
  \fn void set_monoenergy_xray_source(XRaySource const* p_source_manager, float const monoenergy)
  \param p_source_manager - pointer on the source
  \param monoenergy - monoenergetic value
  \brief Set the monoenergy value for the XRaySource
*/
extern "C" GGEMS_EXPORT void set_monoenergy_xray_source(
  XRaySource* p_source_manager, float const monoenergy);

/*!
  \fn void set_polyenergy_xray_source(XRaySource const* p_source_manager, char const* energy_spectrum)
  \param p_source_manager - pointer on the source
  \param energy_spectrum - polyenergetic spectrum
  \brief Set the polyenergetic spectrum value for the XRaySource
*/
extern "C" GGEMS_EXPORT void set_polyenergy_xray_source(
  XRaySource* p_source_manager, char const* energy_spectrum);

#endif // End of GUARD_GGEMS_SOURCES_XRAYSOURCE_HH
