#ifndef GUARD_GGEMS_SOURCES_GGEMSXRAYSOURCE_HH
#define GUARD_GGEMS_SOURCES_GGEMSXRAYSOURCE_HH

/*!
  \file GGEMSXRaySource.hh

  \brief This class define a XRay source in GGEMS useful for CT/CBCT simulation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday October 22, 2019
*/

//#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/sources/GGEMSSource.hh"

/*!
  \class GGEMSXRaySource
  \brief This class define a XRay source in GGEMS useful for CT/CBCT simulation
*/
class GGEMS_EXPORT GGEMSXRaySource : public GGEMSSource
{
  public:
    /*!
      \brief GGEMSXRaySource constructor
    */
    GGEMSXRaySource(void);

    /*!
      \brief GGEMSXRaySource destructor
    */
    ~GGEMSXRaySource(void);

 // public: // Pure abstract method from GGEMSSourceManager
    /*!
      \fn void GetPrimaries(GGulong const& number_of_particles)
      \brief Generate primary particles
    */
 //   void GetPrimaries(GGulong const& number_of_particles);

    /*!
      \fn void Initialize(void)
      \brief Initialize a GGEMS source
    */
 //   void Initialize(void);

    /*!
      \fn void PrintInfos(void) override
      \brief Printing infos about the source
    */
    void PrintInfos(void) const override;

 // public:
    /*!
      \fn void SetBeamAperture(GGfloat const& beam_aperture)
      \param beam_aperture - beam aperture of the x-ray source
      \brief Set the beam aperture of the source
    */
  //  void SetBeamAperture(GGfloat const& beam_aperture);

    /*!
      \fn void SetFocalSpotSize(GGfloat const& width, GGfloat const& height, GGfloat const& depth)
      \param width - width of the focal spot size
      \param height - height of the focal spot size
      \param depth - depth of the focal spot size
      \brief Set the focal spot size of the x-ray source
    */
 //   void SetFocalSpotSize(GGfloat const& width, GGfloat const& height,
 //     GGfloat const& depth);

    /*!
      \fn void SetMonoenergy(GGfloat const& monoenergy)
      \param monoenergy - Monoenergy value
      \brief set the value of energy in monoenergy mode
    */
  //  void SetMonoenergy(GGfloat const& monoenergy);

    /*!
      \fn void SetPolyenergy(char const* energy_spectrum_filename)
      \param energy_spectrum_filename - filename containing the energy spectrum
      \brief set the energy spectrum file for polyenergy mode
    */
 //   void SetPolyenergy(char const* energy_spectrum_filename);

 // private:
    /*!
      \fn void CheckParameters(void) const
      \brief Check mandatory parameters for a X-Ray source
    */
  //  void CheckParameters(void) const;

    /*!
      \fn void FillEnergy(void)
      \brief fill energy for poly or mono energy mode
    */
 //   void FillEnergy(void);

    /*!
      \fn void InitializeKernel(void)
      \brief Initialize kernel for GGEMSXRaySource in OpenCL
    */
  //  void InitializeKernel(void);

  public:
    /*!
      \fn inline GGEMSSource* Clone(void) const = 0
      \brief Clone class to store it in source manager
    */
    inline GGEMSXRaySource* Clone() const override
    {
      return new GGEMSXRaySource(*this);
    }

  private: // Specific members for GGEMSXRaySource
    GGfloat beam_aperture_; /*!< Beam aperture of the x-ray source */
    GGfloat3 focal_spot_size_; /*!< Focal spot size of the x-ray source */
    GGbool is_monoenergy_mode_; /*!< Boolean checking the mode of energy */
    GGfloat monoenergy_; /*!< Monoenergy mode */
    std::string energy_spectrum_filename_; /*!< The energy spectrum filename for polyenergetic mode */
    GGuint number_of_energy_bins_; /*!< Number of energy bins for the polyenergetic mode */

  private: // Buffer for OpenCL
    cl::Buffer* p_energy_spectrum_; /*!< Energy spectrum for OpenCL device */
    cl::Buffer* p_cdf_; /*!< Cumulative distribution function to generate a random energy */
};

/*!
  \fn GGEMSXRaySource* create_ggems_xray_source(void)
  \brief Get the GGEMSXRaySource pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSXRaySource* create_ggems_xray_source(void);

/*!
  \fn void initialize_ggems_xray_source(GGEMSXRaySource* source_manager)
  \param source_manager - pointer on the source
  \brief Initialize the X-Ray source
*/
//extern "C" GGEMS_EXPORT void initialize_ggems_xray_source(
  //GGEMSXRaySource* p_source_manager);

/*!
  \fn void set_position_ggems_xray_source(GGEMSXRaySource* p_source_manager, GGfloat const pos_x, GGfloat const pos_y, GGfloat const pos_z)
  \param source_manager - pointer on the source
  \param pos_x - Position of the source in X
  \param pos_y - Position of the source in Y
  \param pos_z - Position of the source in Z
  \brief Set the position of the source in the global coordinates
*/
//extern "C" GGEMS_EXPORT void set_position_ggems_xray_source(
  //GGEMSXRaySource* p_source_manager, GGfloat const pos_x, GGfloat const pos_y,
  //GGfloat const pos_z);

/*!
  \fn void print_infos_ggems_xray_source(GGEMSXRaySource const* p_source_manager)
  \param p_source_manager - pointer on the source
  \brief Print infos about the GGEMSXRaySource
*/
extern "C" GGEMS_EXPORT void print_infos_ggems_xray_source(
  GGEMSXRaySource* p_source_manager);

/*!
  \fn void set_source_particle_type_ggems_xray_source(GGEMSXRaySource* p_source_manager, char const* particle_name)
  \param source_manager - pointer on the source
  \param particle_name - name/type of the particle: photon or electron
  \brief Set the type of the source particle
*/
extern "C" GGEMS_EXPORT void set_source_particle_type_ggems_xray_source(
  GGEMSXRaySource* p_source_manager, char const* particle_name);

/*!
  \fn void set_beam_aperture_ggems_xray_source(GGEMSXRaySource* p_source_manager, GGfloat const beam_aperture)
  \param p_source_manager - pointer on the source
  \param beam_aperture - beam aperture of the x-ray source
  \brief set the beam aperture of the x-ray source
*/
//extern "C" GGEMS_EXPORT void set_beam_aperture_ggems_xray_source(
  //GGEMSXRaySource* p_source_manager, GGfloat const beam_aperture);

/*!
  \fn void set_focal_spot_size_ggems_xray_source(GGEMSXRaySource* p_source_manager, GGfloat const width, GGfloat const height, GGfloat const depth)
  \param p_source_manager - pointer on the source
  \param width - width of the focal spot size
  \param height - height of the focal spot size
  \param depth - depth of the focal spot size
  \brief Set the focal spot size of the x-ray source
*/
//extern "C" GGEMS_EXPORT void set_focal_spot_size_ggems_xray_source(
  //GGEMSXRaySource* p_source_manager, GGfloat const width, GGfloat const height,
  //GGfloat const depth);

/*!
  \fn void set_local_axis_xray_source(GGEMSXRaySource* p_source_manager, GGfloat const m00, GGfloat const m01, GGfloat const m02, GGfloat const m10, GGfloat const m11, GGfloat const m12, GGfloat const m20, GGfloat const m21, GGfloat const m22)
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
//extern "C" GGEMS_EXPORT void set_local_axis_ggems_xray_source(
  //GGEMSXRaySource* p_source_manager,
  //GGfloat const m00, GGfloat const m01, GGfloat const m02,
  //GGfloat const m10, GGfloat const m11, GGfloat const m12,
  //GGfloat const m20, GGfloat const m21, GGfloat const m22);

/*!
  \fn void set_rotation_xray_source(GGEMSXRaySource* p_source_manager, GGfloat const rx, GGfloat const ry, GGfloat const rz)
  \param p_source_manager - pointer on the source
  \param rx - Rotation around X along global axis
  \param ry - Rotation around Y along global axis
  \param rz - Rotation around Z along global axis
  \brief Set the rotation of the source around global axis
*/
//extern "C" GGEMS_EXPORT void set_rotation_ggems_xray_source(
  //GGEMSXRaySource* p_source_manager, GGfloat const rx, GGfloat const ry,
  //GGfloat const rz);

/*!
  \fn void update_rotation_xray_source(GGEMSXRaySource* p_source_manager, GGfloat const rx, GGfloat const ry, GGfloat const rz)
  \param p_source_manager - pointer on the source
  \param rx - Rotation around X along global axis
  \param ry - Rotation around Y along global axis
  \param rz - Rotation around Z along global axis
  \brief Update the rotation of the source around global axis
*/
//extern "C" GGEMS_EXPORT void update_rotation_ggems_xray_source(
  //GGEMSXRaySource* p_source_manager, GGfloat const rx, GGfloat const ry,
  //GGfloat const rz);

/*!
  \fn void set_monoenergy_ggems_xray_source(GGEMSXRaySource const* p_source_manager, GGfloat const monoenergy)
  \param p_source_manager - pointer on the source
  \param monoenergy - monoenergetic value
  \brief Set the monoenergy value for the GGEMSXRaySource
*/
//extern "C" GGEMS_EXPORT void set_monoenergy_ggems_xray_source(
  //GGEMSXRaySource* p_source_manager, GGfloat const monoenergy);

/*!
  \fn void set_polyenergy_ggems_xray_source(GGEMSXRaySource const* p_source_manager, char const* energy_spectrum)
  \param p_source_manager - pointer on the source
  \param energy_spectrum - polyenergetic spectrum
  \brief Set the polyenergetic spectrum value for the GGEMSXRaySource
*/
//extern "C" GGEMS_EXPORT void set_polyenergy_ggems_xray_source(
  //GGEMSXRaySource* p_source_manager, char const* energy_spectrum);

#endif // End of GUARD_GGEMS_SOURCES_GGEMSXRAYSOURCE_HH
