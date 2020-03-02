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

#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "GGEMS/global/GGEMSExport.hh"

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
      \fn void LoadPhantomImage(std::string const& phantom_filename)
      \param phantom_filename - name of the MHF file containing the phantom
      \brief load phantom image to GGEMS
    */
    void LoadPhantomImage(std::string const& phantom_filename);

    /*!
      \fn void LoadRangeToMaterialData(std::string const& range_data_filename)
      \param range_data_filename - name of the file containing the range to material data
      \brief create a volume of label in GGEMS
    */
    void LoadRangeToMaterialData(std::string const& range_data_filename);

  private:
    std::shared_ptr<cl::Buffer> solid_phantom_data_; /*!< Data about solid phantom */
    std::shared_ptr<cl::Buffer> label_data_; /*!< Pointer storing the buffer about label data */
};

#endif // End of GUARD_GGEMS_GEOMETRIES_GGEMSSOLIDPHANTOM_HH
