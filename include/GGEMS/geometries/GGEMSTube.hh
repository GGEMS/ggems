#ifndef GUARD_GGEMS_GEOMETRY_GGEMSTUBE_HH
#define GUARD_GGEMS_GEOMETRY_GGEMSTUBE_HH

/*!
  \file GGEMSTube.hh

  \brief Class GGEMSTube inheriting from GGEMSVolumeSolid handling Tube solid

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday January 13, 2020
*/

#include "GGEMS/geometries/GGEMSVolumeSolid.hh"

class GGEMS_EXPORT GGEMSTube : public GGEMSVolumeSolid
{
  public:
    /*!
      \brief GGEMSTube constructor
    */
    GGEMSTube(void);

    /*!
      \brief GGEMSTube destructor
    */
    ~GGEMSTube(void);

  public:
    /*!
      \fn GGEMSTube(GGEMSTube const& tube) = delete
      \param tube - reference on the tube solid volume
      \brief Avoid copy of the class by reference
    */
    GGEMSTube(GGEMSTube const& tube) = delete;

    /*!
      \fn GGEMSTube& operator=(GGEMSTube const& tube) = delete
      \param tube - reference on the tube solid volume
      \brief Avoid assignement of the class by reference
    */
    GGEMSTube& operator=(GGEMSTube const& tube) = delete;

    /*!
      \fn GGEMSTube(GGEMSTube const&& tube) = delete
      \param tube - rvalue reference on the tube solid volume
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSTube(GGEMSTube const&& tube) = delete;

    /*!
      \fn GGEMSSourceDefinition& operator=(GGEMSSourceDefinition const&& tube) = delete
      \param tube - rvalue reference on the tube solid volume
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSTube& operator=(GGEMSTube const&& tube) = delete;

  public:
    void SetHeight(GGdouble const& height);
    void SetRadius(GGdouble const& radius);

  public:
    /*!
      \fn void Initialize(void) override
      \brief Initialize the solid and store it in Phantom creator manager
    */
    void Initialize(void) override;

    /*!
      \fn void Draw(void)
      \brief Draw analytical volume in voxelized phantom
    */
    void Draw(void) override;

  protected:
    /*!
      \fn void CheckParameters(void) const
      \brief check parameters for each type of volume
    */
    void CheckParameters(void) const override;

  private:
    GGdouble height_; /*!< Height of the cylinder */
    GGdouble radius_; /*!< Radius of the cylinder */
};

/*!
  \fn GGEMSTube* create_tube(void)
  \brief Create instance of GGEMSTube
*/
extern "C" GGEMS_EXPORT GGEMSTube* create_tube(void);

/*!
  \fn GGEMSTube* create_tube(void)
  \brief Delete instance of GGEMSTube
*/
extern "C" GGEMS_EXPORT void delete_tube(GGEMSTube* p_tube);

/*!
  \fn void void set_height_tube(GGEMSTube* p_tube, GGdouble const height)
  \param p_tube - pointer on the solid tube
  \param height - height of the tube
  \brief Set the height of the tube
*/
extern "C" GGEMS_EXPORT void set_height_tube(GGEMSTube* p_tube,
  GGdouble const height);

/*!
  \fn void void set_radius_tube(GGEMSTube* p_tube, GGdouble const radius)
  \param p_tube - pointer on the solid tube
  \param radius - radius of the tube
  \brief Set the radius of the tube
*/
extern "C" GGEMS_EXPORT void set_radius_tube(GGEMSTube* p_tube,
  GGdouble const radius);

/*!
  \fn void void set_position_tube(GGEMSTube* p_tube, GGdouble const pos_x, GGdouble const pos_y, GGdouble const pos_z)
  \param p_tube - pointer on the solid tube
  \param pos_x - radius of the tube
  \param pos_y - radius of the tube
  \param pos_z - radius of the tube
  \brief Set the position of the tube
*/
extern "C" GGEMS_EXPORT void set_position_tube(GGEMSTube* p_tube,
  GGdouble const pos_x, GGdouble const pos_y, GGdouble const pos_z);

/*!
  \fn void void set_label_value_tube(GGEMSTube* p_tube, GGfloat const label_value)
  \param p_tube - pointer on the solid tube
  \param label_value - label value in tube
  \brief Set the label value in tube
*/
extern "C" GGEMS_EXPORT void set_label_value_tube(GGEMSTube* p_tube,
  GGfloat const label_value);

/*!
  \fn void initialize_tube(GGEMSTube* p_tube)
  \brief Initialize the solid and store it in Phantom creator manager
*/
extern "C" GGEMS_EXPORT void initialize_tube(GGEMSTube* p_tube);

/*!
  \fn void draw_tube(GGEMSTube* p_tube)
  \brief Draw analytical volume in voxelized phantom
*/
extern "C" GGEMS_EXPORT void draw_tube(GGEMSTube* p_tube);

#endif // End of GUARD_GGEMS_GEOMETRY_GGEMSTUBE_HH
