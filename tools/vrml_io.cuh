// GGEMS Copyright (C) 2015

/*!
 * \file vrml_io.cuh
 * \brief VRML IO
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date Monday May 30, 2016
 *
 */

#ifndef VRML_IO_CUH
#define VRML_IO_CUH

#include "global.cuh"

#include "primitives.cuh"

// Sources
#include "beamlet_source.cuh"
#include "cone_beam_CT_source.cuh"

// Phantoms
#include "vox_phan_dosi_nav.cuh"
#include "vox_phan_img_nav.cuh"
#include "vox_phan_iort_nav.cuh"
#include "mesh_phan_linac_nav.cuh"

// Detectors
#include "ct_detector.cuh"

/*!
  \class VRML IO
  \brief This class allows to export GGEMS simulation object (source, phantom, etc) in a 3D representation VRML format file
*/
class VrmlIO
{
public:
    /*!
     * \brief VrmlIO contructor
     */
    VrmlIO() {}

    /*!
     * \brief VrmlIO destructor
     */
    ~VrmlIO() {}

    /*!
     * \fn void open( std::string filename )
     * \brief Open a VRML file to export a GGEMS scene
     * \param filename Name of the VRML file to export
     */
    void open( std::string filename );

    /*!
     * \fn void close()
     * \brief Close the VRML file
     */
    void close();

    /*!
     * \fn void draw_source( BeamletSource* aSource )
     * \brief Draw a Beamlet GGEMS source into the VRML file
     * \param aSource A beamlet source
     */
    void draw_source( BeamletSource* aSource );

    /*!
     * \fn void draw_source( ConeBeamCTSource* aSource )
     * \brief Draw a ConeBeamCT GGEMS source into the VRML file
     * \param aSource The source
     */
    void draw_source( ConeBeamCTSource* aSource );

    /*!
     * \fn void draw_phantom( VoxPhanDosiNav* aPhantom )
     * \brief Draw in 3D a VoxPhanDosiNav into the VRML file
     * \param aPhantom The phantom
     */
    void draw_phantom( VoxPhanDosiNav* aPhantom );

    void draw_phantom( VoxPhanIORTNav* aPhantom );

    void draw_phantom( VoxPhanImgNav* aPhantom );

    void draw_phantom( MeshPhanLINACNav* aPhantom );

    void draw_detector( CTDetector* aDetector );

private:
    FILE *m_pfile;                          /*!< File to export the 3D VRML scene */

private:
    f32xyz m_yellow = {1.0, 1.0, 0.0};      /*!< Pre-defined color: yellow */
    f32xyz m_red = {1.0, 0.0, 0.0};         /*!< Pre-defined color: red */
    f32xyz m_blue = {0.0, 0.0, 1.0};        /*!< Pre-defined color: blue */
    f32xyz m_green = {0.0, 1.0, 0.0};       /*!< Pre-defined color: green */
    f32xyz m_cyan = {0.0, 1.0, 1.0};        /*!< Pre-defined color: cyan */

    /*!
     * \fn void m_draw_point( f32xyz pos, f32xyz color )
     * \brief Private function that draw a point in VRML
     * \param pos 3D position of the point
     * \param color Point color in (r, g, b)
     */
    void m_draw_point( f32xyz pos, f32xyz color );

    /*!
     * \fn void m_draw_sphere( f32xyz pos, f32 radius, f32xyz color, f32 transparency = 0.0 )
     * \brief Private function that draw a sphere in VRML
     * \param pos 3D position of the sphere center
     * \param radius Radius of the sphere
     * \param color Color in (r, g, b) of the sphere
     * \param transparency Transparency of the sphere
     */
    void m_draw_sphere( f32xyz pos, f32 radius, f32xyz color, f32 transparency = 0.0 );

    void m_draw_cone( f32xyz pos, f32xyz angles, f32 height, f32 bottom_radius, f32xyz color, f32 transparency = 0.0 );

    void m_draw_mesh( f32xyz *v1, f32xyz *v2, f32xyz *v3, ui32 nb_tri, f32xyz color, f32 transparency = 0.0, bool inv_normal = false );

//    /*!
//     * \fn void m_draw_obb( ObbData obb, f32xyz color, f32 transparency = 0.0 )
//     * \brief Private function that draw an OBB in VRML
//     * \param obb OBB data object
//     * \param color Color in (r, g, b) of the OBB
//     * \param transparency Transparency of the OBB
//     */
//    void m_draw_obb( ObbData obb, f32xyz color, f32 transparency = 0.0 );

    /*!
     * \fn void m_draw_aabb( AabbData aabb, f32xyz color, f32 transparency = 0.0 )
     * \brief Private function that draw an AABB in VRML
     * \param aabb AABB data object
     * \param color Color in (r, g, b) of the AABB
     * \param transparency Transparency of the AABB
     */
    void m_draw_aabb( AabbData aabb, f32xyz color, f32 transparency = 0.0 );

    /*!
     * \fn void m_draw_wireframe_aabb( AabbData aabb, f32xyz color )
     * \brief Private function that draw in wireframe an AABB in VRML
     * \param aabb AABB data object
     * \param color Color in (r, g, b) of the AABB
     */
    void m_draw_wireframe_aabb( AabbData aabb, f32xyz color );

    void m_draw_wireframe_obb( ObbData obb, f32matrix44 trans, f32xyz color );

    void m_draw_wireframe_cone( f32xyz pos, f32 aperture, f32matrix44 trans, f32xyz color );

    /*!
     * \fn void m_draw_axis( f32xyz org, f32xyz ax, f32xyz ay, f32xyz az )
     * \brief Private function that draw axis in VRML
     * \param org Position of the origin
     * \param ax Vector of the x-axis
     * \param ay Vector of the y-axis
     * \param az Vector of the z-axis
     */
    void m_draw_axis( f32xyz org, f32xyz ax, f32xyz ay, f32xyz az );

};

#endif
