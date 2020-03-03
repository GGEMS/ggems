#ifndef GUARD_GGEMS_GEOMETRIES_GGEMSSOLIDPHANTOM_HH
#define GUARD_GGEMS_GEOMETRIES_GGEMSSOLIDPHANTOM_HH

/*!
  \file GGEMSSolidPhantom.hh

  \brief GGEMS class for solid phantom. This class reads the phantom volume infos, the range data file

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 2, 2020
*/

#include <string>
#include <memory>
#include <fstream>

#include "GGEMS/global/GGEMSOpenCLManager.hh"
#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSTools.hh"

/*!
  \class GGEMSSolidPhantom
  \brief GGEMS class for solid phantom informations
*/
class GGEMS_EXPORT GGEMSSolidPhantom
{
  public:
    /*!
      \brief GGEMSSolidPhantom constructor
    */
    GGEMSSolidPhantom(void);

    /*!
      \brief GGEMSSolidPhantom destructor
    */
    ~GGEMSSolidPhantom(void);

    /*!
      \fn GGEMSSolidPhantom(GGEMSSolidPhantom const& solid_phantom) = delete
      \param solid_phantom - reference on the GGEMS solid phantom
      \brief Avoid copy by reference
    */
    GGEMSSolidPhantom(GGEMSSolidPhantom const& solid_phantom) = delete;

    /*!
      \fn GGEMSSolidPhantom& operator=(GGEMSSolidPhantom const& solid_phantom) = delete
      \param solid_phantom - reference on the GGEMS solid phantom
      \brief Avoid assignement by reference
    */
    GGEMSSolidPhantom& operator=(GGEMSSolidPhantom const& solid_phantom) = delete;

    /*!
      \fn GGEMSSolidPhantom(GGEMSSolidPhantom const&& solid_phantom) = delete
      \param solid_phantom - rvalue reference on the GGEMS solid phantom
      \brief Avoid copy by rvalue reference
    */
    GGEMSSolidPhantom(GGEMSSolidPhantom const&& solid_phantom) = delete;

    /*!
      \fn GGEMSSolidPhantom& operator=(GGEMSSolidPhantom const&& solid_phantom) = delete
      \param solid_phantom - rvalue reference on the GGEMS solid phantom
      \brief Avoid copy by rvalue reference
    */
    GGEMSSolidPhantom& operator=(GGEMSSolidPhantom const&& solid_phantom) = delete;

    /*!
      \fn void LoadPhantomImage(std::string const& phantom_filename, std::string const& range_data_filename)
      \param phantom_filename - name of the MHF file containing the phantom
      \param range_data_filename - name of the file containing the range to material data
      \brief load phantom image to GGEMS and create a volume of label in GGEMS
    */
    void LoadPhantomImage(std::string const& phantom_filename, std::string const& range_data_filename);

    /*!
      \fn void ApplyOffset(GGdouble3 const& offset_xyz)
      \param offset_xyz - offset in X, Y and Z
      \brief apply an offset defined by the user
    */
    void ApplyOffset(GGdouble3 const& offset_xyz);

    /*!
      \fn void PrintInfos(void) const
      \brief printing infos about solid phantom
    */
    void PrintInfos(void) const;

  private:
    /*!
      \fn template <typename T> void ConvertImageToLabel(std::string const& raw_data_filename)
      \tparam T - type of data
      \param raw_data_filename - raw data filename from mhd
      \brief convert image data to label data
    */
    template <typename T>
    void ConvertImageToLabel(std::string const& raw_data_filename);

  private:
    std::shared_ptr<cl::Buffer> solid_phantom_data_; /*!< Data about solid phantom */
    std::shared_ptr<cl::Buffer> label_data_; /*!< Pointer storing the buffer about label data */
    GGEMSOpenCLManager& opencl_manager_; /*!< Reference to OpenCL manager singleton */
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename T>
void GGEMSSolidPhantom::ConvertImageToLabel(std::string const& raw_data_filename)
{
  GGcout("GGEMSSolidPhantom", "ConvertImageToLabel", 3) << "Converting image material data to label data..." << GGendl;

  // Checking if file exists
  std::ifstream in_raw_stream(raw_data_filename, std::ios::in | std::ios::binary);
  GGEMSFileStream::CheckInputStream(in_raw_stream, raw_data_filename);

  // Get pointer on OpenCL device
  GGEMSSolidPhantomData* solid_data = opencl_manager_.GetDeviceBuffer<GGEMSSolidPhantomData>(solid_phantom_data_, sizeof(GGEMSSolidPhantomData));

  // Get information about mhd file
  GGuint const kNumberOfVoxels = solid_data->number_of_voxels_;

  // Release the pointer
  opencl_manager_.ReleaseDeviceBuffer(solid_phantom_data_, solid_data);

  // Reading data to a tmp buffer
  std::unique_ptr<T[]> tmp_raw_data = std::make_unique<T[]>(kNumberOfVoxels);
  in_raw_stream.read(reinterpret_cast<char*>(tmp_raw_data.get()), kNumberOfVoxels * sizeof(T));

  // Closing file
  in_raw_stream.close();
}

#endif // End of GUARD_GGEMS_GEOMETRIES_GGEMSSOLIDPHANTOM_HH
