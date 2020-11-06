#ifndef GUARD_GGEMS_SOURCES_GGEMSXRAYSOURCE_HH
#define GUARD_GGEMS_SOURCES_GGEMSXRAYSOURCE_HH

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
  \file GGEMSXRaySource.hh

  \brief This class define a XRay source in GGEMS useful for CT/CBCT simulation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday October 22, 2019
*/

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
    explicit GGEMSXRaySource(std::string const& source_name);

    /*!
      \brief GGEMSXRaySource destructor
    */
    ~GGEMSXRaySource(void);

    /*!
      \fn GGEMSXRaySource(GGEMSXRaySource const& xray_source) = delete
      \param xray_source - reference on the GGEMS XRay source
      \brief Avoid copy by reference
    */
    GGEMSXRaySource(GGEMSXRaySource const& xray_source) = delete;

    /*!
      \fn GGEMSXRaySource& operator=(GGEMSXRaySource const& xray_source) = delete
      \param xray_source - reference on the GGEMS XRay source
      \brief Avoid assignement by reference
    */
    GGEMSXRaySource& operator=(GGEMSXRaySource const& xray_source) = delete;

    /*!
      \fn GGEMSXRaySource(GGEMSXRaySource const&& xray_source) = delete
      \param xray_source - rvalue reference on the GGEMS XRay source
      \brief Avoid copy by rvalue reference
    */
    GGEMSXRaySource(GGEMSXRaySource const&& xray_source) = delete;

    /*!
      \fn GGEMSXRaySource& operator=(GGEMSXRaySource const&& xray_source) = delete
      \param xray_source - rvalue reference on the GGEMS XRay source
      \brief Avoid copy by rvalue reference
    */
    GGEMSXRaySource& operator=(GGEMSXRaySource const&& xray_source) = delete;

    /*!
      \fn void SetBeamAperture(GGfloat const& beam_aperture, std::string const& unit)
      \param beam_aperture - beam aperture of the x-ray source
      \param unit - unit of the angle
      \brief Set the beam aperture of the source
    */
    void SetBeamAperture(GGfloat const& beam_aperture, std::string const& unit = "deg");

    /*!
      \fn void SetFocalSpotSize(GGfloat const& width, GGfloat const& height, GGfloat const& depth, std::string const& unit)
      \param width - width of the focal spot size
      \param height - height of the focal spot size
      \param depth - depth of the focal spot size
      \param unit - unit of the distance
      \brief Set the focal spot size of the x-ray source
    */
    void SetFocalSpotSize(GGfloat const& width, GGfloat const& height, GGfloat const& depth, std::string const& unit = "mm");

    /*!
      \fn void SetMonoenergy(GGfloat const& monoenergy, std::string const& unit)
      \param monoenergy - Monoenergy value
      \param unit - unit of the energy
      \brief set the value of energy in monoenergy mode
    */
    void SetMonoenergy(GGfloat const& monoenergy, std::string const& unit = "keV");

    /*!
      \fn void SetPolyenergy(std::string const& energy_spectrum_filename)
      \param energy_spectrum_filename - filename containing the energy spectrum
      \brief set the energy spectrum file for polyenergy mode
    */
    void SetPolyenergy(std::string const& energy_spectrum_filename);

    /*!
      \fn void Initialize(void)
      \brief Initialize a GGEMS source
    */
    void Initialize(void) override;

    /*!
      \fn void PrintInfos(void) const
      \brief Printing infos about the source
    */
    void PrintInfos(void) const override;

    /*!
      \fn void GetPrimaries(GGlong const& number_of particles) = 0
      \param number_of_particles - number of particles to generate
      \brief Generate primary particles
    */
    void GetPrimaries(GGlong const& number_of_particles) override;

  private:
    /*!
      \fn void InitializeKernel(void)
      \brief Initialize kernel for specific source in OpenCL
    */
    void InitializeKernel(void) override;

    /*!
      \fn void FillEnergy(void)
      \brief fill energy for poly or mono energy mode
    */
    void FillEnergy(void);

    /*!
      \fn void CheckParameters(void) const
      \brief Check mandatory parameters for a source
    */
    void CheckParameters(void) const override;

  private: // Specific members for GGEMSXRaySource
    GGfloat beam_aperture_; /*!< Beam aperture of the x-ray source */
    GGfloat3 focal_spot_size_; /*!< Focal spot size of the x-ray source */
    GGbool is_monoenergy_mode_; /*!< Boolean checking the mode of energy */
    GGfloat monoenergy_; /*!< Monoenergy mode */
    std::string energy_spectrum_filename_; /*!< The energy spectrum filename for polyenergetic mode */
    GGint number_of_energy_bins_; /*!< Number of energy bins for the polyenergetic mode */
    std::shared_ptr<cl::Buffer> energy_spectrum_cl_; /*!< Energy spectrum for OpenCL device */
    std::shared_ptr<cl::Buffer> cdf_cl_; /*!< Cumulative distribution function to generate a random energy */
};

/*!
  \fn GGEMSXRaySource* create_ggems_xray_source(char const* source_name)
  \return the pointer on the singleton
  \param source_name - name of the source
  \brief Get the GGEMSXRaySource pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSXRaySource* create_ggems_xray_source(char const* source_name);

/*!
  \fn void initialize_ggems_xray_source(GGEMSXRaySource* xray_source)
  \param xray_source - pointer on the source
  \brief Initialize the X-Ray source
*/
extern "C" GGEMS_EXPORT void initialize_ggems_xray_source(GGEMSXRaySource* xray_source);

/*!
  \fn void set_position_ggems_xray_source(GGEMSXRaySource* xray_source, GGfloat const pos_x, GGfloat const pos_y, GGfloat const pos_z, char const* unit)
  \param xray_source - pointer on the source
  \param pos_x - Position of the source in X
  \param pos_y - Position of the source in Y
  \param pos_z - Position of the source in Z
  \param unit - unit of the distance
  \brief Set the position of the source in the global coordinates
*/
extern "C" GGEMS_EXPORT void set_position_ggems_xray_source(GGEMSXRaySource* xray_source, GGfloat const pos_x, GGfloat const pos_y, GGfloat const pos_z, char const* unit);

/*!
  \fn void set_number_of_particles_xray_source(GGEMSXRaySource* xray_source, GGlong const number_of_particles)
  \param xray_source - pointer on the source
  \param number_of_particles - number of particles to simulate
  \brief Set the number of particles to simulate during the simulation
*/
extern "C" GGEMS_EXPORT void set_number_of_particles_xray_source(GGEMSXRaySource* xray_source, GGlong const number_of_particles);

/*!
  \fn void set_source_particle_type_ggems_xray_source(GGEMSXRaySource* xray_source, char const* particle_name)
  \param xray_source - pointer on the source
  \param particle_name - name/type of the particle: photon or electron
  \brief Set the type of the source particle
*/
extern "C" GGEMS_EXPORT void set_source_particle_type_ggems_xray_source(GGEMSXRaySource* xray_source, char const* particle_name);

/*!
  \fn void set_beam_aperture_ggems_xray_source(GGEMSXRaySource* xray_source, GGfloat const beam_aperture, char const* unit)
  \param xray_source - pointer on the source
  \param beam_aperture - beam aperture of the x-ray source
  \param unit - unit of the angle
  \brief set the beam aperture of the x-ray source
*/
extern "C" GGEMS_EXPORT void set_beam_aperture_ggems_xray_source(GGEMSXRaySource* xray_source, GGfloat const beam_aperture, char const* unit);

/*!
  \fn void set_focal_spot_size_ggems_xray_source(GGEMSXRaySource* xray_source, GGfloat const width, GGfloat const height, GGfloat const depth, char const* unit)
  \param xray_source - pointer on the source
  \param width - width of the focal spot size
  \param height - height of the focal spot size
  \param depth - depth of the focal spot size
  \param unit - unit of the distance
  \brief Set the focal spot size of the x-ray source
*/
extern "C" GGEMS_EXPORT void set_focal_spot_size_ggems_xray_source(GGEMSXRaySource* xray_source, GGfloat const width, GGfloat const height, GGfloat const depth, char const* unit);

/*!
  \fn void set_local_axis_ggems_xray_source(GGEMSXRaySource* xray_source, GGfloat const m00, GGfloat const m01, GGfloat const m02, GGfloat const m10, GGfloat const m11, GGfloat const m12, GGfloat const m20, GGfloat const m21, GGfloat const m22)
  \param xray_source - pointer on the source
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
extern "C" GGEMS_EXPORT void set_local_axis_ggems_xray_source(GGEMSXRaySource* xray_source, GGfloat const m00, GGfloat const m01, GGfloat const m02, GGfloat const m10, GGfloat const m11, GGfloat const m12, GGfloat const m20, GGfloat const m21, GGfloat const m22);

/*!
  \fn void set_rotation_ggems_xray_source(GGEMSXRaySource* xray_source, GGfloat const rx, GGfloat const ry, GGfloat const rz, char const* unit)
  \param xray_source - pointer on the source
  \param rx - Rotation around X along global axis
  \param ry - Rotation around Y along global axis
  \param rz - Rotation around Z along global axis
  \param unit - unit of the degree
  \brief Set the rotation of the source around global axis
*/
extern "C" GGEMS_EXPORT void set_rotation_ggems_xray_source(GGEMSXRaySource* xray_source, GGfloat const rx, GGfloat const ry, GGfloat const rz, char const* unit);

/*!
  \fn void set_monoenergy_ggems_xray_source(GGEMSXRaySource* xray_source, GGfloat const monoenergy, char const* unit)
  \param xray_source - pointer on the source
  \param monoenergy - monoenergetic value
  \param unit - unit of the energy
  \brief Set the monoenergy value for the GGEMSXRaySource
*/
extern "C" GGEMS_EXPORT void set_monoenergy_ggems_xray_source(GGEMSXRaySource* xray_source, GGfloat const monoenergy, char const* unit);

/*!
  \fn void set_polyenergy_ggems_xray_source(GGEMSXRaySource* xray_source, char const* energy_spectrum)
  \param xray_source - pointer on the source
  \param energy_spectrum - polyenergetic spectrum
  \brief Set the polyenergetic spectrum value for the GGEMSXRaySource
*/
extern "C" GGEMS_EXPORT void set_polyenergy_ggems_xray_source(GGEMSXRaySource* xray_source, char const* energy_spectrum);

#endif // End of GUARD_GGEMS_SOURCES_GGEMSXRAYSOURCE_HH
