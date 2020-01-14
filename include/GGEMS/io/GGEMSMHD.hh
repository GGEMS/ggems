#ifndef GUARD_GGEMS_IO_GGEMSMHD_HH
#define GUARD_GGEMS_IO_GGEMSMHD_HH

/*!
  \file GGEMSMHD.hh

  \brief I/O class handling MHD file

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

/*!
  \class GGEMSMHD
  \brief I/O class handling MHD file
*/
class GGEMS_EXPORT GGEMSMHD
{
  public:
    /*!
      \brief GGEMSMHD constructor
    */
    GGEMSMHD(void);

    /*!
      \brief GGEMSMHD destructor
    */
    ~GGEMSMHD(void);

  public:
    /*!
      \fn GGEMSMHD(GGEMSMHD const& mhd) = delete
      \param mhd - reference on the mhd file
      \brief Avoid copy of the class by reference
    */
    GGEMSMHD(GGEMSMHD const& mhd) = delete;

    /*!
      \fn GGEMSMHD& operator=(GGEMSMHD const& mhd) = delete
      \param mhd - reference on the mhd file
      \brief Avoid assignement of the class by reference
    */
    GGEMSMHD& operator=(GGEMSMHD const& mhd) = delete;

    /*!
      \fn GGEMSMHD(GGEMSMHD const&& mhd) = delete
      \param mhd - rvalue reference on the mhd file
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSMHD(GGEMSMHD const&& mhd) = delete;

    /*!
      \fn GGEMSMHD& operator=(GGEMSMHD const&& mhd) = delete
      \param mhd - rvalue reference on the mhd file
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSMHD& operator=(GGEMSMHD const&& mhd) = delete;

  public:
    /*!
      \fn void SetBaseName(std::string const& basename)
      \param basename - basename of the mhd file
      \brief set the basename for mhd header/raw file
    */
    void SetBaseName(std::string const& basename);

    /*!
      \fn void Write(void)
      \brief Write mhd header/raw file
    */
    void Write(void);

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
};

#endif // End of GUARD_GGEMS_IO_GGEMSMHD_HH
