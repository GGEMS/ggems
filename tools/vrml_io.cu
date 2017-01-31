// GGEMS Copyright (C) 2017

/*!
 * \file vrml_io.cuh
 * \brief VRML IO
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date Monday May 30, 2016
 *
 */

#ifndef VRML_IO_CU
#define VRML_IO_CU


#include "vrml_io.cuh"

// ================ Main functions ===================================

void VrmlIO::open( std::string filename )
{
    // Get ext
    std::string ext = filename.substr( filename.find_last_of( "." ) + 1 );

    // Get lower case
    std::transform( ext.begin(), ext.end(), ext.begin(), ::tolower );

    // Check extension
    if ( ext != "wrl" )
    {
        GGwarn << "GGEMS have changed the extension of the VRML file to '.wrl'" << GGendl;

        std::string filename_noext = filename.substr( 0, filename.find_last_of ( "." ) );
        filename = filename_noext + ".wrl";
    }

    // Open the file
    m_pfile = fopen( filename.c_str(), "w" );

    // Check opening
    if ( !m_pfile )
    {
        GGcerr << "Error, file " << filename << " impossible to open!" << GGendl;
        exit_simulation();
    }

    // Write the header
    fprintf( m_pfile, "#VRML V2.0 utf8\n" );
    fprintf( m_pfile, "# Exported from GGEMS\n\n" );

//    fprintf( m_pfile, "Viewpoint {\n" );
//    fprintf( m_pfile, "  position    0 0 0\n");
//    fprintf( m_pfile, "  orientation 0 0 -1 0\n");
//    fprintf( m_pfile, "  description 'view all'\n");
//    fprintf( m_pfile, "}\n\n");

    // Main Axis
    f32xyz org = { 0.0, 0.0, 0.0 };
    f32xyz ax = { 20.0, 0.0, 0.0 };
    f32xyz ay = { 0.0, 20.0, 0.0 };
    f32xyz az = { 0.0, 0.0, 20.0 };
    m_draw_axis( org, ax, ay, az );

}

void VrmlIO::close()
{
    fclose( m_pfile );
}

void VrmlIO::draw_source( BeamletSource *aSource )
{
    // Get back information from this source
    f32xyz loc_pos = aSource->get_local_beamlet_position();
    f32xyz loc_src = aSource->get_local_source_position();
    f32xyz loc_size = aSource->get_local_size();
    f32matrix44 trans = aSource->get_transformation_matrix();

    // Src
    f32xyz gbl_src = fxyz_local_to_global_position( trans, loc_src );
    m_draw_sphere( gbl_src, 5, m_yellow );

    // local Axis
    f32xyz org = { 0.0,  0.0,  0.0 };
    f32xyz ax = { 20.0,  0.0,  0.0 };
    f32xyz ay = {  0.0, 20.0,  0.0 };
    f32xyz az = {  0.0,  0.0, 20.0 };
    org = fxyz_local_to_global_position( trans, org );
    ax = fxyz_local_to_global_position( trans, ax );
    ay = fxyz_local_to_global_position( trans, ay );
    az = fxyz_local_to_global_position( trans, az );
    m_draw_axis( org, ax, ay, az );

    // Get distance to draw the line that connect the gbl source and phantom
    //
    //                  + gbl src
    //                 /|
    //            E   / |   A
    //               /  |
    //      loc pos +   + org
    //             /    |
    //         F  /     |   B     C=A+B   G=E+F
    //           /      |
    //    ? --> +       + isocenter
    //
    f32 dist_A = fxyz_mag( fxyz_sub( org, gbl_src ) );
    f32 dist_C = fxyz_mag( gbl_src );
    f32xyz delta_E = fxyz_sub( fxyz_local_to_global_position( trans, loc_pos ) , gbl_src );
    f32 dist_E = fxyz_mag( delta_E );
    f32 dist_G = dist_E * dist_C / dist_A; // Thales

    // Compute the position (first get a direction)
    f32xyz dir = fxyz_unit( delta_E );
    f32xyz end_pos = fxyz_add( gbl_src, fxyz_scale( dir, dist_G ) );

    // draw beamlet
    //
    //          + Src
    //
    //
    //     a+-----+b      In a generic way (meanning any direction of the beamlet is considered for drawing),
    //     /     /|       the 2D beamlet is a 3D oriented cube that one dimension is equal to zero.
    //   d+-----+c|
    //    |e+   | +f
    //    |     |/
    //   h+-----+g
    //

    f32xyz hsize = fxyz_scale( loc_size, 0.5f );

    f32xyz a = { loc_pos.x-hsize.x, loc_pos.y-hsize.y, loc_pos.z-hsize.z };
    f32xyz b = { loc_pos.x+hsize.x, loc_pos.y-hsize.y, loc_pos.z-hsize.z };
    f32xyz c = { loc_pos.x+hsize.x, loc_pos.y+hsize.y, loc_pos.z-hsize.z };
    f32xyz d = { loc_pos.x-hsize.x, loc_pos.y+hsize.y, loc_pos.z-hsize.z };

    f32xyz e = { loc_pos.x-hsize.x, loc_pos.y-hsize.y, loc_pos.z+hsize.z };
    f32xyz f = { loc_pos.x+hsize.x, loc_pos.y-hsize.y, loc_pos.z+hsize.z };
    f32xyz g = { loc_pos.x+hsize.x, loc_pos.y+hsize.y, loc_pos.z+hsize.z };
    f32xyz h = { loc_pos.x-hsize.x, loc_pos.y+hsize.y, loc_pos.z+hsize.z };

    a = fxyz_local_to_global_position( trans, a );
    b = fxyz_local_to_global_position( trans, b );
    c = fxyz_local_to_global_position( trans, c );
    d = fxyz_local_to_global_position( trans, d );
    e = fxyz_local_to_global_position( trans, e );
    f = fxyz_local_to_global_position( trans, f );
    g = fxyz_local_to_global_position( trans, g );
    h = fxyz_local_to_global_position( trans, h );

    fprintf( m_pfile, "\n# Beamlet\n" );
    fprintf( m_pfile, "Shape {\n");

    fprintf( m_pfile, "  geometry IndexedLineSet {\n");
    // Coordinate
    fprintf( m_pfile, "    coord Coordinate {\n");
    fprintf( m_pfile, "      point [\n");
    fprintf( m_pfile, "        %f %f %f,\n", a.x, a.y, a.z); // a 0
    fprintf( m_pfile, "        %f %f %f,\n", b.x, b.y, b.z); // b 1
    fprintf( m_pfile, "        %f %f %f,\n", c.x, c.y, c.z); // c 2
    fprintf( m_pfile, "        %f %f %f,\n", d.x, d.y, d.z); // d 3
    fprintf( m_pfile, "        %f %f %f,\n", e.x, e.y, e.z); // e 4
    fprintf( m_pfile, "        %f %f %f,\n", f.x, f.y, f.z); // f 5
    fprintf( m_pfile, "        %f %f %f,\n", g.x, g.y, g.z); // g 6
    fprintf( m_pfile, "        %f %f %f,\n", h.x, h.y, h.z); // h 7
    fprintf( m_pfile, "        %f %f %f,\n", gbl_src.x, gbl_src.y, gbl_src.z); // gbl_src 8
    fprintf( m_pfile, "        %f %f %f,\n", end_pos.x, end_pos.y, end_pos.z); // end_pos 9
    fprintf( m_pfile, "      ]\n");
    fprintf( m_pfile, "    }\n");
    // CoordIndex
    fprintf( m_pfile, "    coordIndex [\n");
    fprintf( m_pfile, "      %i, %i, %i, %i, %i, -1,\n", 0, 1, 2, 3, 0); // top
    fprintf( m_pfile, "      %i, %i, %i, %i, %i, -1,\n", 4, 5, 6, 7, 4); // bottom
    fprintf( m_pfile, "      %i, %i, -1,\n", 0, 4);
    fprintf( m_pfile, "      %i, %i, -1,\n", 1, 5);
    fprintf( m_pfile, "      %i, %i, -1,\n", 2, 6);
    fprintf( m_pfile, "      %i, %i, -1,\n", 3, 7);
    fprintf( m_pfile, "      %i, %i, -1,\n", 8, 9);
    fprintf( m_pfile, "    ]\n");

    // Color
    fprintf( m_pfile, "    color Color {\n");
    fprintf( m_pfile, "      color [%f %f %f]\n", m_yellow.x, m_yellow.y, m_yellow.z);
    fprintf( m_pfile, "    }\n");
    fprintf( m_pfile, "    colorIndex [0, 0, 0, 0, 0, 0, 0]\n");
    fprintf( m_pfile, "    colorPerVertex FALSE\n");

    fprintf( m_pfile, "  }\n");
    fprintf( m_pfile, "}\n");



}

void VrmlIO::draw_source( ConeBeamCTSource *aSource )
{
    // Get back information from this source
    f32xyz pos = aSource->get_position();
    f32 aperture = aSource->get_aperture();
    f32matrix44 trans = aSource->get_transformation_matrix();

    m_draw_wireframe_cone( pos, aperture, trans, m_yellow );

    // local Axis
    f32xyz org = { 1.0, 1.0, 1.0 }; // Small shift of the axis for a better visibility (overlap with obb)
    f32xyz ax = { 20.0, 1.0, 1.0 };
    f32xyz ay = { 1.0, 20.0, 1.0 };
    f32xyz az = { 1.0, 1.0, 20.0 };
    org = fxyz_local_to_global_position( trans, org );
    ax = fxyz_local_to_global_position( trans, ax );
    ay = fxyz_local_to_global_position( trans, ay );
    az = fxyz_local_to_global_position( trans, az );
    m_draw_axis( org, ax, ay, az );
}

void VrmlIO::draw_source( PointSource *aSource )
{
    f32xyz pos = aSource->get_position();

//    m_draw_point( pos, m_yellow );
    m_draw_sphere( pos, 2.0, m_yellow );
}

void VrmlIO::draw_phantom( VoxPhanDosiNav *aPhantom )
{
    // Get back some info
    AabbData aabb = aPhantom->get_bounding_box();

    m_draw_wireframe_aabb( aabb, m_blue );
}

void VrmlIO::draw_phantom( VoxPhanIORTNav *aPhantom )
{
    // Get back some info
    AabbData aabb = aPhantom->get_bounding_box();

    m_draw_wireframe_aabb( aabb, m_blue );
}

void VrmlIO::draw_phantom( VoxPhanImgNav *aPhantom )
{
    // Get back some info
    AabbData aabb = aPhantom->get_bounding_box();

    m_draw_wireframe_aabb( aabb, m_blue );
}

void VrmlIO::draw_phantom( MeshPhanLINACNav *aPhantom )
{
    // Get the geometry
    const LinacData *linac = aPhantom->get_linac_geometry();
    f32matrix44 trans = aPhantom->get_linac_transformation();

//    GGcout << "Matrix: " << GGendl;
//    printf("%f %f %f %f\n", trans.m00, trans.m01, trans.m02, trans.m03);
//    printf("%f %f %f %f\n", trans.m10, trans.m11, trans.m12, trans.m13);
//    printf("%f %f %f %f\n", trans.m20, trans.m21, trans.m22, trans.m23);
//    printf("%f %f %f %f\n", trans.m30, trans.m31, trans.m32, trans.m33);

    // local Axis
    f32xyz org = { 0.0,  0.0,  0.0 };
    f32xyz ax = { 20.0,  0.0,  0.0 };
    f32xyz ay = {  0.0, 20.0,  0.0 };
    f32xyz az = {  0.0,  0.0, 20.0 };
    org = fxyz_local_to_global_position( trans, org );
    ax = fxyz_local_to_global_position( trans, ax );
    ay = fxyz_local_to_global_position( trans, ay );
    az = fxyz_local_to_global_position( trans, az );
    m_draw_axis( org, ax, ay, az );

    ui32 ileaf;

    /// Tranform all leaves ///////////////////////////////////////////////
    ileaf = 0; while ( ileaf < linac->A_nb_leaves ) // linac->A_nb_leaves
    {
        ui32 offset = linac->A_leaf_index[ ileaf ];
        ui32 nbtri = linac->A_leaf_nb_triangles[ ileaf ];

        ui32 itri = 0; while ( itri < nbtri )
        {
            linac->A_leaf_v1[ offset + itri ] = fxyz_local_to_global_position( trans, linac->A_leaf_v1[ offset + itri ] );
            linac->A_leaf_v2[ offset + itri ] = fxyz_local_to_global_position( trans, linac->A_leaf_v2[ offset + itri ] );
            linac->A_leaf_v3[ offset + itri ] = fxyz_local_to_global_position( trans, linac->A_leaf_v3[ offset + itri ] );
            ++itri;
        }
        ++ileaf;
    }

    ileaf = 0; while ( ileaf < linac->B_nb_leaves )
    {
        ui32 offset = linac->B_leaf_index[ ileaf ];
        ui32 nbtri = linac->B_leaf_nb_triangles[ ileaf ];

        ui32 itri = 0; while ( itri < nbtri )
        {
            linac->B_leaf_v1[ offset + itri ] = fxyz_local_to_global_position( trans, linac->B_leaf_v1[ offset + itri ] );
            linac->B_leaf_v2[ offset + itri ] = fxyz_local_to_global_position( trans, linac->B_leaf_v2[ offset + itri ] );
            linac->B_leaf_v3[ offset + itri ] = fxyz_local_to_global_position( trans, linac->B_leaf_v3[ offset + itri ] );
            ++itri;
        }
        ++ileaf;
    }

    /// Transform backup ///////////////////////////////////////////////////

    ui32 offset = linac->X_jaw_index[ 0 ];
    ui32 nbtri = linac->X_jaw_nb_triangles[ 0 ];
    ui32 itri = 0; while ( itri < nbtri )
    {
        linac->X_jaw_v1[ offset + itri ] = fxyz_local_to_global_position( trans, linac->X_jaw_v1[ offset + itri ] );
        linac->X_jaw_v2[ offset + itri ] = fxyz_local_to_global_position( trans, linac->X_jaw_v2[ offset + itri ] );
        linac->X_jaw_v3[ offset + itri ] = fxyz_local_to_global_position( trans, linac->X_jaw_v3[ offset + itri ] );
        ++itri;
    }

    offset = linac->X_jaw_index[ 1 ];
    nbtri = linac->X_jaw_nb_triangles[ 1 ];
    itri = 0; while ( itri < nbtri )
    {
        linac->X_jaw_v1[ offset + itri ] = fxyz_local_to_global_position( trans, linac->X_jaw_v1[ offset + itri ] );
        linac->X_jaw_v2[ offset + itri ] = fxyz_local_to_global_position( trans, linac->X_jaw_v2[ offset + itri ] );
        linac->X_jaw_v3[ offset + itri ] = fxyz_local_to_global_position( trans, linac->X_jaw_v3[ offset + itri ] );
        ++itri;
    }

    offset = linac->Y_jaw_index[ 0 ];
    nbtri = linac->Y_jaw_nb_triangles[ 0 ];
    itri = 0; while ( itri < nbtri )
    {
        linac->Y_jaw_v1[ offset + itri ] = fxyz_local_to_global_position( trans, linac->Y_jaw_v1[ offset + itri ] );
        linac->Y_jaw_v2[ offset + itri ] = fxyz_local_to_global_position( trans, linac->Y_jaw_v2[ offset + itri ] );
        linac->Y_jaw_v3[ offset + itri ] = fxyz_local_to_global_position( trans, linac->Y_jaw_v3[ offset + itri ] );
        ++itri;
    }

    offset = linac->Y_jaw_index[ 1 ];
    nbtri = linac->Y_jaw_nb_triangles[ 1 ];
    itri = 0; while ( itri < nbtri )
    {
        linac->Y_jaw_v1[ offset + itri ] = fxyz_local_to_global_position( trans, linac->Y_jaw_v1[ offset + itri ] );
        linac->Y_jaw_v2[ offset + itri ] = fxyz_local_to_global_position( trans, linac->Y_jaw_v2[ offset + itri ] );
        linac->Y_jaw_v3[ offset + itri ] = fxyz_local_to_global_position( trans, linac->Y_jaw_v3[ offset + itri ] );
        ++itri;
    }

    /// Drawing //////////////////////////////////////////////////////////////

    // Draw leaves from bank A
    ileaf = 0; while ( ileaf < linac->A_nb_leaves )    // linac->A_nb_leaves
    {
        ui32 offset = linac->A_leaf_index[ ileaf ];
        ui32 nbtri = linac->A_leaf_nb_triangles[ ileaf ];

        if ( ileaf == 0 )
        {
            m_draw_mesh( &(linac->A_leaf_v1[ offset ]), &(linac->A_leaf_v2[ offset ]), &(linac->A_leaf_v3[ offset ]),
                         nbtri, m_green, 0.0, false );
        }
        else
        {
            m_draw_mesh( &(linac->A_leaf_v1[ offset ]), &(linac->A_leaf_v2[ offset ]), &(linac->A_leaf_v3[ offset ]),
                         nbtri, m_blue, 0.0, false );
        }



        ++ileaf;
    }

    // Draw leaves from bank B
    ileaf = 0; while ( ileaf < linac->B_nb_leaves )    // linac->A_nb_leaves
    {
        ui32 offset = linac->B_leaf_index[ ileaf ];
        ui32 nbtri = linac->B_leaf_nb_triangles[ ileaf ];

        m_draw_mesh( &(linac->B_leaf_v1[ offset ]), &(linac->B_leaf_v2[ offset ]), &(linac->B_leaf_v3[ offset ]),
                     nbtri, m_red, 0.0, true );

        ++ileaf;
    }

    // Draw jaws
    if ( linac->X_nb_jaw != 0 )
    {
        m_draw_mesh( &(linac->X_jaw_v1[ linac->X_jaw_index[ 0 ] ]),
                     &(linac->X_jaw_v2[ linac->X_jaw_index[ 0 ] ]),
                     &(linac->X_jaw_v3[ linac->X_jaw_index[ 0 ] ]),
                     linac->X_jaw_nb_triangles[ 0 ], m_blue, 0.0, true );

        m_draw_mesh( &(linac->X_jaw_v1[ linac->X_jaw_index[ 1 ] ]),
                     &(linac->X_jaw_v2[ linac->X_jaw_index[ 1 ] ]),
                     &(linac->X_jaw_v3[ linac->X_jaw_index[ 1 ] ]),
                     linac->X_jaw_nb_triangles[ 1 ], m_red, 0.0, false );
    }

    if ( linac->Y_nb_jaw != 0 )
    {
        m_draw_mesh( &(linac->Y_jaw_v1[ linac->Y_jaw_index[ 0 ] ]),
                     &(linac->Y_jaw_v2[ linac->Y_jaw_index[ 0 ] ]),
                     &(linac->Y_jaw_v3[ linac->Y_jaw_index[ 0 ] ]),
                     linac->Y_jaw_nb_triangles[ 0 ], m_blue, 0.0, false );

        m_draw_mesh( &(linac->Y_jaw_v1[ linac->Y_jaw_index[ 1 ] ]),
                     &(linac->Y_jaw_v2[ linac->Y_jaw_index[ 1 ] ]),
                     &(linac->Y_jaw_v3[ linac->Y_jaw_index[ 1 ] ]),
                     linac->Y_jaw_nb_triangles[ 1 ], m_red, 0.0, true );
    }


//    // Draw bounding box
//    m_draw_wireframe_aabb( linac->aabb, m_cyan );
//    m_draw_wireframe_aabb( linac->A_bank_aabb, m_cyan );
//    m_draw_wireframe_aabb( linac->B_bank_aabb, m_cyan );
//    m_draw_wireframe_aabb( linac->X_jaw_aabb[ 0 ], m_cyan );
//    m_draw_wireframe_aabb( linac->X_jaw_aabb[ 1 ], m_cyan );
//    m_draw_wireframe_aabb( linac->Y_jaw_aabb[ 0 ], m_cyan );
//    m_draw_wireframe_aabb( linac->Y_jaw_aabb[ 1 ], m_cyan );


}

void VrmlIO::draw_detector( CTDetector *aDetector )
{
    // Get back info
    ObbData obb = aDetector->get_bounding_box();
    f32matrix44 trans = aDetector->get_transformation();

    m_draw_wireframe_obb( obb, trans, m_green );

    // local Axis
    f32xyz org = { 1.0, 1.0, 1.0 }; // Small shift of the axis for a better visibility (overlap with obb)
    f32xyz ax = { 20.0, 1.0, 1.0 };
    f32xyz ay = { 1.0, 20.0, 1.0 };
    f32xyz az = { 1.0, 1.0, 20.0 };
    org = fxyz_local_to_global_position( trans, org );
    ax = fxyz_local_to_global_position( trans, ax );
    ay = fxyz_local_to_global_position( trans, ay );
    az = fxyz_local_to_global_position( trans, az );
    m_draw_axis( org, ax, ay, az );

}

// =========== Private functions ==========================================

void VrmlIO::m_draw_point( f32xyz pos, f32xyz color )
{
    fprintf( m_pfile, "\n# Point\n" );
    fprintf( m_pfile, "Shape {\n" );
    fprintf( m_pfile, "  geometry PointSet {\n" );
    fprintf( m_pfile, "    coord Coordinate {\n" );
    fprintf( m_pfile, "      point [ %f %f %f ]\n", pos.x, pos.y, pos.z );
    fprintf( m_pfile, "    }\n" );
    fprintf( m_pfile, "    color Color {\n" );
    fprintf( m_pfile, "      color [ %f %f %f ]\n", color.x, color.y, color.z );
    fprintf( m_pfile, "    }\n" );
    fprintf( m_pfile, "  }\n" );
    fprintf( m_pfile, "}\n" );
}

void VrmlIO::m_draw_sphere( f32xyz pos, f32 radius, f32xyz color, f32 transparency )
{
    fprintf( m_pfile, "\n# Sphere\n" );
    fprintf( m_pfile, "Transform {\n" );
    fprintf( m_pfile, "  translation %f %f %f\n", pos.x, pos.y, pos.z );
    fprintf( m_pfile, "  children [\n" );
    fprintf( m_pfile, "     Shape {\n" );
    fprintf( m_pfile, "        geometry Sphere {\n" );
    fprintf( m_pfile, "          radius %f \n", radius );
    fprintf( m_pfile, "        }\n" );
    fprintf( m_pfile, "        appearance Appearance {\n" );
    fprintf( m_pfile, "           material Material {\n" );
    fprintf( m_pfile, "              diffuseColor %f %f %f \n", color.x, color.y, color.z );
    fprintf( m_pfile, "              transparency %f\n", transparency );
    fprintf( m_pfile, "           }\n" );
    fprintf( m_pfile, "        }\n" );
    fprintf( m_pfile, "     }\n" );
    fprintf( m_pfile, "  ]\n" );
    fprintf( m_pfile, "}\n" );
}

/* TODO: to review according to the new obb format
void VrmlIO::m_draw_obb( ObbData obb, f32xyz color, f32 transparency )
{

    fprintf( m_pfile, "\n# OBB\n" );
    fprintf( m_pfile, "Transform {\n");
    fprintf( m_pfile, "  translation %f %f %f\n", obb.translate.x, obb.translate.y, obb.translate.z );
    fprintf( m_pfile, "  rotation 1.0 0.0 0.0 %f\n", obb.angle.x /deg);  // Must be in degree
    fprintf( m_pfile, "  rotation 0.0 1.0 0.0 %f\n", obb.angle.y /deg);
    fprintf( m_pfile, "  rotation 0.0 0.0 1.0 %f\n", obb.angle.z /deg);
    fprintf( m_pfile, "  children [\n");
    fprintf( m_pfile, "    Shape {\n");
    fprintf( m_pfile, "      appearance Appearance {\n");
    fprintf( m_pfile, "        material Material {\n");
    fprintf( m_pfile, "          diffuseColor %f %f %f\n", color.x, color.y, color.z);
    fprintf( m_pfile, "          transparency %f\n", transparency);
    fprintf( m_pfile, "        }\n");
    fprintf( m_pfile, "      }\n");
    fprintf( m_pfile, "      geometry Box {\n");
    fprintf( m_pfile, "        size %f %f %f\n", obb.xmax-obb.xmin, obb.ymax-obb.ymin, obb.zmax-obb.zmin);
    fprintf( m_pfile, "      }\n");
    fprintf( m_pfile, "    }\n");
    fprintf( m_pfile, "  ]\n");
    fprintf( m_pfile, "}\n");

}
*/

void VrmlIO::m_draw_cone(f32xyz pos, f32xyz angles, f32 height, f32 bottom_radius, f32xyz color, f32 transparency )
{
    fprintf( m_pfile, "\n# Cone\n" );
    fprintf( m_pfile, "Transform {\n");
    fprintf( m_pfile, "  translation %f %f %f\n", pos.x, pos.y, pos.z );
    fprintf( m_pfile, "  rotation 1.0 0.0 0.0 %f\n", angles.x /deg);  // Must be in degree
    fprintf( m_pfile, "  rotation 0.0 1.0 0.0 %f\n", angles.y /deg);
    fprintf( m_pfile, "  rotation 0.0 0.0 1.0 %f\n", angles.z /deg);
    fprintf( m_pfile, "  children [\n");
    fprintf( m_pfile, "    Shape {\n");
    fprintf( m_pfile, "      appearance Appearance {\n");
    fprintf( m_pfile, "        material Material {\n");
    fprintf( m_pfile, "          diffuseColor %f %f %f\n", color.x, color.y, color.z);
    fprintf( m_pfile, "          transparency %f\n", transparency);
    fprintf( m_pfile, "        }\n");
    fprintf( m_pfile, "      }\n");
    fprintf( m_pfile, "      geometry Cone {\n");
    fprintf( m_pfile, "        bottomRadius %f\n", bottom_radius);
    fprintf( m_pfile, "        height %f\n", height);
    fprintf( m_pfile, "        side TRUE\n");
    fprintf( m_pfile, "        bottom TRUE\n");
    fprintf( m_pfile, "      }\n");
    fprintf( m_pfile, "    }\n");
    fprintf( m_pfile, "  ]\n");
    fprintf( m_pfile, "}\n");
}

void VrmlIO::m_draw_aabb( AabbData aabb, f32xyz color, f32 transparency )
{
    f32 tx = 0.5f * ( aabb.xmax + aabb.xmin );
    f32 ty = 0.5f * ( aabb.ymax + aabb.ymin );
    f32 tz = 0.5f * ( aabb.zmax + aabb.zmin );

    fprintf( m_pfile, "\n# AABB\n" );
    fprintf( m_pfile, "Transform {\n");
    fprintf( m_pfile, "  translation %f %f %f\n", tx, ty, tz );
    fprintf( m_pfile, "  children [\n");
    fprintf( m_pfile, "    Shape {\n");
    fprintf( m_pfile, "      appearance Appearance {\n");
    fprintf( m_pfile, "        material Material {\n");
    fprintf( m_pfile, "          diffuseColor %f %f %f\n", color.x, color.y, color.z);
    fprintf( m_pfile, "          transparency %f\n", transparency);
    fprintf( m_pfile, "        }\n");
    fprintf( m_pfile, "      }\n");
    fprintf( m_pfile, "      geometry Box {\n");
    fprintf( m_pfile, "        size %f %f %f\n", aabb.xmax-aabb.xmin, aabb.ymax-aabb.ymin, aabb.zmax-aabb.zmin);
    fprintf( m_pfile, "      }\n");
    fprintf( m_pfile, "    }\n");
    fprintf( m_pfile, "  ]\n");
    fprintf( m_pfile, "}\n");
}

void VrmlIO::m_draw_mesh(f32xyz *v1, f32xyz *v2, f32xyz *v3, ui32 nb_tri, f32xyz color, f32 transparency , bool inv_normal)
{
    fprintf( m_pfile, "\n# MESH\n" );
    fprintf( m_pfile, "Shape {\n" );
    fprintf( m_pfile, "  appearance Appearance {\n" );
    fprintf( m_pfile, "    material Material {\n" );
    fprintf( m_pfile, "      diffuseColor %f %f %f\n", color.x, color.y, color.z );
    fprintf( m_pfile, "      transparency %f\n", transparency );
    fprintf( m_pfile, "    }\n" );
    fprintf( m_pfile, "  }\n" );

    fprintf( m_pfile, "  geometry IndexedFaceSet {\n" );
    fprintf( m_pfile, "    coord Coordinate {\n" );
    fprintf( m_pfile, "      point [\n" );
    ui32 i=0; while (i < nb_tri) {
        fprintf( m_pfile, "        %f %f %f,\n", v1[ i ].x, v1[ i ].y, v1[ i ].z );
        fprintf( m_pfile, "        %f %f %f,\n", v2[ i ].x, v2[ i ].y, v2[ i ].z );
        fprintf( m_pfile, "        %f %f %f,\n", v3[ i ].x, v3[ i ].y, v3[ i ].z );
        ++i;
    }
    fprintf( m_pfile, "      ]\n" );
    fprintf( m_pfile, "    }\n" );
    fprintf( m_pfile, "    coordIndex [\n" );

    if ( inv_normal )
    {
        i=0; while ( i < nb_tri ) {
            ui32 ind = 3*i;
            fprintf( m_pfile, "      %i, %i, %i, -1,\n", ind+2, ind+1, ind );
            ++i;
        }
    }
    else
    {
        i=0; while ( i < nb_tri ) {
            ui32 ind = 3*i;
            fprintf( m_pfile, "      %i, %i, %i, -1,\n", ind, ind+1, ind+2 );
            ++i;
        }
    }

    fprintf( m_pfile, "    ]\n" );
    fprintf( m_pfile, "  }\n" );
    fprintf( m_pfile, "}\n" );
}

void VrmlIO::m_draw_wireframe_aabb(AabbData aabb, f32xyz color )
{

    //          xmin        xmax
    //          3+---------2+
    //          /          /|
    //         /          / |
    // ymin  0+---------1+  |
    //        |          |  |
    //        | 7+       | 6+   zmax
    //        |          | /
    //        |          |/
    // ymax  4+---------5+   zmin

    fprintf( m_pfile, "\n# AABB Wireframe\n" );
    fprintf( m_pfile, "Shape {\n");

    fprintf( m_pfile, "  geometry IndexedLineSet {\n");
    // Coordinate
    fprintf( m_pfile, "    coord Coordinate {\n");
    fprintf( m_pfile, "      point [\n");
    fprintf( m_pfile, "        %f %f %f,\n", aabb.xmin, aabb.ymin, aabb.zmin); // 0
    fprintf( m_pfile, "        %f %f %f,\n", aabb.xmax, aabb.ymin, aabb.zmin); // 1
    fprintf( m_pfile, "        %f %f %f,\n", aabb.xmax, aabb.ymin, aabb.zmax); // 2
    fprintf( m_pfile, "        %f %f %f,\n", aabb.xmin, aabb.ymin, aabb.zmax); // 3
    fprintf( m_pfile, "        %f %f %f,\n", aabb.xmin, aabb.ymax, aabb.zmin); // 4
    fprintf( m_pfile, "        %f %f %f,\n", aabb.xmax, aabb.ymax, aabb.zmin); // 5
    fprintf( m_pfile, "        %f %f %f,\n", aabb.xmax, aabb.ymax, aabb.zmax); // 6
    fprintf( m_pfile, "        %f %f %f,\n", aabb.xmin, aabb.ymax, aabb.zmax); // 7
    fprintf( m_pfile, "      ]\n");
    fprintf( m_pfile, "    }\n");
    // CoordIndex
    fprintf( m_pfile, "    coordIndex [\n");
    fprintf( m_pfile, "      %i, %i, %i, %i, %i, -1,\n", 0, 1, 2, 3, 0); // top
    fprintf( m_pfile, "      %i, %i, %i, %i, %i, -1,\n", 4, 5, 6, 7, 4); // bottom
    fprintf( m_pfile, "      %i, %i, -1,\n", 0, 4);
    fprintf( m_pfile, "      %i, %i, -1,\n", 1, 5);
    fprintf( m_pfile, "      %i, %i, -1,\n", 2, 6);
    fprintf( m_pfile, "      %i, %i, -1,\n", 3, 7);
    fprintf( m_pfile, "    ]\n");

    // Color
    fprintf( m_pfile, "    color Color {\n");
    fprintf( m_pfile, "      color [%f %f %f]\n", color.x, color.y, color.z);
    fprintf( m_pfile, "    }\n");
    fprintf( m_pfile, "    colorIndex [0, 0, 0, 0, 0, 0]\n");
    fprintf( m_pfile, "    colorPerVertex FALSE\n");

    fprintf( m_pfile, "  }\n");
    fprintf( m_pfile, "}\n");

}

void VrmlIO::m_draw_wireframe_obb( ObbData obb, f32matrix44 trans, f32xyz color )
{
    // Compute the vertice
    //          xmin        xmax
    //          3+---------2+
    //          /          /|
    //         /          / |
    // ymin  0+---------1+  |
    //        |          |  |
    //        | 7+       | 6+   zmax
    //        |          | /
    //        |          |/
    // ymax  4+---------5+   zmin

    f32xyz p0 = fxyz_local_to_global_position( trans, make_f32xyz( obb.xmin, obb.ymin, obb.zmin ) );
    f32xyz p1 = fxyz_local_to_global_position( trans, make_f32xyz( obb.xmax, obb.ymin, obb.zmin ) );
    f32xyz p2 = fxyz_local_to_global_position( trans, make_f32xyz( obb.xmax, obb.ymin, obb.zmax ) );
    f32xyz p3 = fxyz_local_to_global_position( trans, make_f32xyz( obb.xmin, obb.ymin, obb.zmax ) );
    f32xyz p4 = fxyz_local_to_global_position( trans, make_f32xyz( obb.xmin, obb.ymax, obb.zmin ) );
    f32xyz p5 = fxyz_local_to_global_position( trans, make_f32xyz( obb.xmax, obb.ymax, obb.zmin ) );
    f32xyz p6 = fxyz_local_to_global_position( trans, make_f32xyz( obb.xmax, obb.ymax, obb.zmax ) );
    f32xyz p7 = fxyz_local_to_global_position( trans, make_f32xyz( obb.xmin, obb.ymax, obb.zmax ) );

    fprintf( m_pfile, "\n# OBB Wireframe\n" );
    fprintf( m_pfile, "Shape {\n");

    fprintf( m_pfile, "  geometry IndexedLineSet {\n");
    // Coordinate
    fprintf( m_pfile, "    coord Coordinate {\n");
    fprintf( m_pfile, "      point [\n");
    fprintf( m_pfile, "        %f %f %f,\n", p0.x, p0.y, p0.z ); // 0
    fprintf( m_pfile, "        %f %f %f,\n", p1.x, p1.y, p1.z ); // 1
    fprintf( m_pfile, "        %f %f %f,\n", p2.x, p2.y, p2.z ); // 2
    fprintf( m_pfile, "        %f %f %f,\n", p3.x, p3.y, p3.z ); // 3
    fprintf( m_pfile, "        %f %f %f,\n", p4.x, p4.y, p4.z ); // 4
    fprintf( m_pfile, "        %f %f %f,\n", p5.x, p5.y, p5.z ); // 5
    fprintf( m_pfile, "        %f %f %f,\n", p6.x, p6.y, p6.z ); // 6
    fprintf( m_pfile, "        %f %f %f,\n", p7.x, p7.y, p7.z ); // 7
    fprintf( m_pfile, "      ]\n");
    fprintf( m_pfile, "    }\n");
    // CoordIndex
    fprintf( m_pfile, "    coordIndex [\n");
    fprintf( m_pfile, "      %i, %i, %i, %i, %i, -1,\n", 0, 1, 2, 3, 0); // top
    fprintf( m_pfile, "      %i, %i, %i, %i, %i, -1,\n", 4, 5, 6, 7, 4); // bottom
    fprintf( m_pfile, "      %i, %i, -1,\n", 0, 4);
    fprintf( m_pfile, "      %i, %i, -1,\n", 1, 5);
    fprintf( m_pfile, "      %i, %i, -1,\n", 2, 6);
    fprintf( m_pfile, "      %i, %i, -1,\n", 3, 7);
    fprintf( m_pfile, "    ]\n");

    // Color
    fprintf( m_pfile, "    color Color {\n");
    fprintf( m_pfile, "      color [%f %f %f]\n", color.x, color.y, color.z);
    fprintf( m_pfile, "    }\n");
    fprintf( m_pfile, "    colorIndex [0, 0, 0, 0, 0, 0]\n");
    fprintf( m_pfile, "    colorPerVertex FALSE\n");

    fprintf( m_pfile, "  }\n");
    fprintf( m_pfile, "}\n");
}

void VrmlIO::m_draw_wireframe_cone( f32xyz pos, f32 aperture, f32matrix44 trans, f32xyz color )
{
    // Compute the vertice
    //
    //                         b0 + + c0  bottom radius
    //  pos                       |/
    //   +------------------------o isocenter
    //          height           /|
    //                       c1 + + b1
    //

//    // Determined the major axis
//    ui8 axis = 0;
//    f32 max = fabs( pos.x );
//    if ( fabs( pos.y ) > max )
//    {
//        max = fabs( pos.y );
//        axis = 1;
//    }
//    if ( fabs( pos.z ) > max )
//    {
//        axis = 2;
//    }

    // Get pos and height
    f32xyz zero = make_f32xyz( 0.0, 0.0, 0.0 );
    pos = fxyz_local_to_global_position( trans, zero );

    f32xyz vec = fxyz_sub( zero, pos );
    f32 height = fxyz_mag( vec );

    // Compute bottom radius
    f32 rad = height * tan( aperture );

    f32xyz a0 = make_f32xyz( 0.0, 0.0, -rad );
    f32xyz a1 = make_f32xyz( 0.0, 0.0,  rad );
    f32xyz b0 = make_f32xyz( 0.0, -rad, 0.0 );
    f32xyz b1 = make_f32xyz( 0.0,  rad, 0.0 );
    f32xyz c0 = make_f32xyz( -rad, 0.0, 0.0 );
    f32xyz c1 = make_f32xyz( rad, 0.0, 0.0 );

    a0 = fxyz_local_to_global_direction( trans, a0 );
    a1 = fxyz_local_to_global_direction( trans, a1 );
    b0 = fxyz_local_to_global_direction( trans, b0 );
    b1 = fxyz_local_to_global_direction( trans, b1 );
    c0 = fxyz_local_to_global_direction( trans, c0 );
    c1 = fxyz_local_to_global_direction( trans, c1 );

    a0 = fxyz_scale( a0, rad );
    a1 = fxyz_scale( a1, rad );
    b0 = fxyz_scale( b0, rad );
    b1 = fxyz_scale( b1, rad );
    c0 = fxyz_scale( c0, rad );
    c1 = fxyz_scale( c1, rad );

//    // Get b0 and b1 point
//    f32xyz b0, b1, c0, c1;
//    if ( axis == 0 )
//    {
//        // Along zy-axis
//        b0 = make_f32xyz( 0.0, 0.0, -rad );
//        b1 = make_f32xyz( 0.0, 0.0,  rad );
//        c0 = make_f32xyz( 0.0, -rad, 0.0 );
//        c1 = make_f32xyz( 0.0,  rad, 0.0 );
//    }
//    else if ( axis == 1 )
//    {
//        // Along x-axis
//        b0 = make_f32xyz( -rad, 0.0, 0.0 );
//        b1 = make_f32xyz(  rad, 0.0, 0.0 );
//        c0 = make_f32xyz( 0.0, 0.0, -rad );
//        c1 = make_f32xyz( 0.0, 0.0,  rad );
//    }
//    else
//    {
//        // Along y-axis
//        b0 = make_f32xyz( 0.0, -rad, 0.0 );
//        b1 = make_f32xyz( 0.0,  rad, 0.0 );
//        c0 = make_f32xyz( -rad, 0.0, 0.0 );
//        c1 = make_f32xyz( rad,  0.0, 0.0 );
//    }

//    b0 = fxyz_local_to_global_position( trans, b0 );
//    b1 = fxyz_local_to_global_position( trans, b1 );
//    c0 = fxyz_local_to_global_position( trans, c0 );
//    c1 = fxyz_local_to_global_position( trans, c1 );

    fprintf( m_pfile, "\n# Cone Wireframe\n" );
    fprintf( m_pfile, "Shape {\n");

    fprintf( m_pfile, "  geometry IndexedLineSet {\n");
    // Coordinate
    fprintf( m_pfile, "    coord Coordinate {\n");
    fprintf( m_pfile, "      point [\n");
    fprintf( m_pfile, "        %f %f %f,\n", pos.x, pos.y, pos.z ); // pos        0
    fprintf( m_pfile, "        %f %f %f,\n", 0.0, 0.0, 0.0 );       // isocenter  1
    fprintf( m_pfile, "        %f %f %f,\n", b0.x, b0.y, b0.z );    // b0         2
    fprintf( m_pfile, "        %f %f %f,\n", b1.x, b1.y, b1.z );    // b1         3
    fprintf( m_pfile, "        %f %f %f,\n", c0.x, c0.y, c0.z );    // c0         4
    fprintf( m_pfile, "        %f %f %f,\n", c1.x, c1.y, c1.z );    // c1         5
    fprintf( m_pfile, "        %f %f %f,\n", a0.x, a0.y, a0.z );    // a0         6
    fprintf( m_pfile, "        %f %f %f,\n", a1.x, a1.y, a1.z );    // a1         7
    fprintf( m_pfile, "      ]\n");
    fprintf( m_pfile, "    }\n");
    // CoordIndex
    fprintf( m_pfile, "    coordIndex [\n");
    fprintf( m_pfile, "      %i, %i, -1,\n", 0, 1);
    fprintf( m_pfile, "      %i, %i, -1,\n", 2, 3);
    fprintf( m_pfile, "      %i, %i, -1,\n", 0, 2);
    fprintf( m_pfile, "      %i, %i, -1,\n", 0, 3);
    fprintf( m_pfile, "      %i, %i, -1,\n", 0, 4);
    fprintf( m_pfile, "      %i, %i, -1,\n", 0, 5);
    fprintf( m_pfile, "      %i, %i, -1,\n", 4, 5);
    fprintf( m_pfile, "      %i, %i, -1,\n", 0, 6);
    fprintf( m_pfile, "      %i, %i, -1,\n", 0, 7);
    fprintf( m_pfile, "      %i, %i, -1,\n", 6, 7);
    fprintf( m_pfile, "    ]\n");

    // Color
    fprintf( m_pfile, "    color Color {\n");
    fprintf( m_pfile, "      color [%f %f %f]\n", color.x, color.y, color.z);
    fprintf( m_pfile, "    }\n");
    fprintf( m_pfile, "    colorIndex [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]\n");
    fprintf( m_pfile, "    colorPerVertex FALSE\n");

    fprintf( m_pfile, "  }\n");
    fprintf( m_pfile, "}\n");


}

void VrmlIO::m_draw_axis( f32xyz org, f32xyz ax, f32xyz ay, f32xyz az )
{
    fprintf( m_pfile, "\n# Axis\n" );
    fprintf( m_pfile, "Shape {\n");

    fprintf( m_pfile, "  geometry IndexedLineSet {\n");
    // Coordinate
    fprintf( m_pfile, "    coord Coordinate {\n");
    fprintf( m_pfile, "      point [\n");
    fprintf( m_pfile, "        %f %f %f,\n", org.x, org.y, org.z); // 0
    fprintf( m_pfile, "        %f %f %f,\n", ax.x, ax.y, ax.z);    // 1
    fprintf( m_pfile, "        %f %f %f,\n", ay.x, ay.y, ay.z);    // 2
    fprintf( m_pfile, "        %f %f %f,\n", az.x, az.y, az.z);    // 3
    fprintf( m_pfile, "      ]\n");
    fprintf( m_pfile, "    }\n");
    // CoordIndex
    fprintf( m_pfile, "    coordIndex [\n");
    fprintf( m_pfile, "      %i, %i, -1,\n", 0, 1); // X
    fprintf( m_pfile, "      %i, %i, -1,\n", 0, 2); // Y
    fprintf( m_pfile, "      %i, %i, -1,\n", 0, 3); // Z
    fprintf( m_pfile, "    ]\n");

    // Color
    fprintf( m_pfile, "    color Color {\n");
    fprintf( m_pfile, "      color [\n");
    fprintf( m_pfile, "        %f %f %f,\n", 1.0, 0.0, 0.0 ); // Red
    fprintf( m_pfile, "        %f %f %f,\n", 0.0, 1.0, 0.0 ); // Green
    fprintf( m_pfile, "        %f %f %f,\n", 0.0, 0.0, 1.0 ); // Blue
    fprintf( m_pfile, "      ]\n");
    fprintf( m_pfile, "    }\n");
    fprintf( m_pfile, "    colorIndex [0, 1, 2]\n");
    fprintf( m_pfile, "    colorPerVertex FALSE\n");

    fprintf( m_pfile, "  }\n");
    fprintf( m_pfile, "}\n");

}








#endif

















