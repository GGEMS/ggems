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

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSTypes.hh"
#include "GGEMS/global/GGEMSOpenCLManager.hh"

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
      \fn void Write(cl::Buffer* p_image) const
      \param p_image - image to write on output file
      \brief Write mhd header/raw file
    */
    void Write(cl::Buffer* p_image) const;

    /*!
      \fn void SetElementSizes(GGdouble3 const& element_sizes)
      \param element_sizes - size of elements in X, Y, Z
      \brief set the size of the elements
    */
    void SetElementSizes(GGdouble3 const& element_sizes);

    /*!
      \fn void SetDimensions(GGuint3 const& dimensions)
      \param dimensions - dimensions of image in X, Y, Z
      \brief set the dimensions of the image
    */
    void SetDimensions(GGuint3 const& dimensions);

    /*!
      \fn void SetOffsets(GGdouble3 const& offsets)
      \param offsets - offset of image in X, Y, Z
      \brief set offsets for the image
    */
    void SetOffsets(GGdouble3 const& offsets);

  private:
    /*!
      \fn void CheckParameters(void) const
      \brief Check parameters before read/write MHD file
    */
    void CheckParameters(void) const;

  private:
    std::string mhd_header_file_; /*!< Name of the MHD header file */
    std::string mhd_raw_file_; /*!< Name of the MHD raw file */
    GGdouble3 element_sizes_; /*!< Size of elements */
    GGuint3 dimensions_; /*!< Dimension volume X, Y, Z */
    GGdouble3 offsets_; /*!< Offset of the image */

  private:
    GGEMSOpenCLManager& opencl_manager_; /*!< Reference to opencl manager singleton */
};

#endif // End of GUARD_GGEMS_IO_GGEMSMHDIMAGE_HH
