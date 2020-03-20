#ifndef GUARD_GGEMS_IO_GGEMSMHDIMAGE_HH
#define GUARD_GGEMS_IO_GGEMSMHDIMAGE_HH

/*!
  \file GGEMSMHDImage.hh

  \brief I/O class handling MHD image file

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday January 14, 2020
*/

#ifdef _MSC_VER
#pragma warning(disable: 4251) // Deleting warning exporting STL members!!!
#endif

#include <string>
#include <memory>

#include "GGEMS/global/GGEMSOpenCLManager.hh"

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSTypes.hh"

/*!
  \class GGEMSMHDImage
  \brief I/O class handling MHD file
*/
class GGEMS_EXPORT GGEMSMHDImage
{
  public:
    /*!
      \brief GGEMSMHDImage constructor
    */
    GGEMSMHDImage(void);

    /*!
      \brief GGEMSMHDImage destructor
    */
    ~GGEMSMHDImage(void);

  public:
    /*!
      \fn GGEMSMHDImage(GGEMSMHDImage const& mhd) = delete
      \param mhd - reference on the mhd file
      \brief Avoid copy of the class by reference
    */
    GGEMSMHDImage(GGEMSMHDImage const& mhd) = delete;

    /*!
      \fn GGEMSMHDImage& operator=(GGEMSMHDImage const& mhd) = delete
      \param mhd - reference on the mhd file
      \brief Avoid assignement of the class by reference
    */
    GGEMSMHDImage& operator=(GGEMSMHDImage const& mhd) = delete;

    /*!
      \fn GGEMSMHDImage(GGEMSMHDImage const&& mhd) = delete
      \param mhd - rvalue reference on the mhd file
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSMHDImage(GGEMSMHDImage const&& mhd) = delete;

    /*!
      \fn GGEMSMHDImage& operator=(GGEMSMHDImage const&& mhd) = delete
      \param mhd - rvalue reference on the mhd file
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSMHDImage& operator=(GGEMSMHDImage const&& mhd) = delete;

  public:
    /*!
      \fn void SetBaseName(std::string const& basename)
      \param basename - basename of the mhd file
      \brief set the basename for mhd header/raw file
    */
    void SetBaseName(std::string const& basename);

    /*!
      \fn void Read(std::string const& image_mhd_header_filename, std::shared_ptr<cl::Buffer> solid_phantom_data)
      \param image_mhd_header_filename - input mhd filename
      \param solid_phantom_data - pointer on solid phantom data
      \brief read the mhd header
    */
    void Read(std::string const& image_mhd_header_filename, std::shared_ptr<cl::Buffer> solid_phantom_data);

    /*!
      \fn void Write(std::shared_ptr<cl::Buffer> image) const
      \param image - image to write on output file
      \brief Write mhd header/raw file
    */
    void Write(std::shared_ptr<cl::Buffer> image) const;

    /*!
      \fn void SetElementSizes(GGfloat3 const& element_sizes)
      \param element_sizes - size of elements in X, Y, Z
      \brief set the size of the elements
    */
    void SetElementSizes(GGfloat3 const& element_sizes);

    /*!
      \fn void SetDimensions(GGuint3 const& dimensions)
      \param dimensions - dimensions of image in X, Y, Z
      \brief set the dimensions of the image
    */
    void SetDimensions(GGuint3 const& dimensions);

    /*!
      \fn void SetDataType(std::string const& data_type)
      \param data_type - type of data
      \brief set the type of data
    */
    void SetDataType(std::string const& data_type);

    /*!
      \fn std::string GetDataMHDType(void) const
      \brief get the mhd data type
      \return the type of data for mhd file
    */
    inline std::string GetDataMHDType(void) const {return mhd_data_type_;};

    /*!
      \fn std::string GetRawMDHfilename(void) const
      \brief get the filename of raw data
      \return the name of raw file
    */
    inline std::string GetRawMDHfilename(void) const {return mhd_raw_file_;};

  private:
    /*!
      \fn void CheckParameters(void) const
      \brief Check parameters before read/write MHD file
    */
    void CheckParameters(void) const;

    /*!
      \fn template <typename T> void WriteRaw(std::shared_ptr<cl::Buffer> image) const
      \tparam T - type of the data
      \param image - image to write on output file
      \brief write the raw data to file
    */
    template <typename T>
    void WriteRaw(std::shared_ptr<cl::Buffer> image) const;

  private:
    std::string mhd_header_file_; /*!< Name of the MHD header file */
    std::string mhd_raw_file_; /*!< Name of the MHD raw file */
    std::string mhd_data_type_; /*!< Type of data */
    GGfloat3 element_sizes_; /*!< Size of elements */
    GGuint3 dimensions_; /*!< Dimension volume X, Y, Z */
    GGEMSOpenCLManager& opencl_manager_; /*!< Reference to opencl manager singleton */
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename T>
void GGEMSMHDImage::WriteRaw(std::shared_ptr<cl::Buffer> image) const
{
  // raw data
  std::ofstream out_raw_stream(mhd_raw_file_, std::ios::out | std::ios::binary);

  // Mapping data
  T* data_image = opencl_manager_.GetDeviceBuffer<T>(image, dimensions_.s[0] * dimensions_.s[1] * dimensions_.s[2] * sizeof(T));

  // Writing data on file
  out_raw_stream.write(reinterpret_cast<char*>(data_image), dimensions_.s[0] * dimensions_.s[1] * dimensions_.s[2] * sizeof(T));

  // Release the pointers
  opencl_manager_.ReleaseDeviceBuffer(image, data_image);
  out_raw_stream.close();
}

#endif // End of GUARD_GGEMS_IO_GGEMSMHDIMAGE_HH
