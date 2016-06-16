// GGEMS Copyright (C) 2015

/*!
 * \file mesh_phan_linac_nav.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date Monday June 13, 2016
 *
 * v0.1: JB - First code
 *
 */

#ifndef MESH_PHAN_LINAC_NAV_CU
#define MESH_PHAN_LINAC_NAV_CU

#include "mesh_phan_linac_nav.cuh"

////// HOST-DEVICE GPU Codes ////////////////////////////////////////////


////// Privates

// Read the list of tokens in a txt line
std::vector< std::string > MeshPhanLINACNav::m_split_txt( std::string line ) {

    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::copy(std::istream_iterator<std::string>(iss),
         std::istream_iterator<std::string>(),
         std::back_inserter(tokens));

    return tokens;

}


void MeshPhanLINACNav::m_init_mlc()
{
    // First check the file
    std::string ext = m_mlc_filename.substr( m_mlc_filename.find_last_of( "." ) + 1 );
    if ( ext != "obj" )
    {
        GGcerr << "MeshPhanLINACNav can only read mesh data in Wavefront format (.obj)!" << GGendl;
        exit_simulation();
    }

    // Then get data
    MeshIO *meshio = new MeshIO;
    MeshData mlc = meshio->read_mesh_file( m_mlc_filename );

//    ui32 i = 0; while ( i < mlc.mesh_names.size() )
//    {
//        GGcout << "Mesh " << i << GGendl;
//        ui32 offset = mlc.mesh_index[ i ];

//        ui32 j = 0; while ( j < mlc.nb_triangles[ i ] )
//        {
//            ui32 ii = offset+j;
//            printf("  %f %f %f  -  %f %f %f  -  %f %f %f\n", mlc.v1[ii].x, mlc.v1[ii].y, mlc.v1[ii].z,
//                   mlc.v2[ii].x, mlc.v2[ii].y, mlc.v2[ii].z,
//                   mlc.v3[ii].x, mlc.v3[ii].y, mlc.v3[ii].z );
//            ++j;
//        }

//        ++i;
//    }

    GGcout << "Meshes read" << GGendl;

    // Check if there are at least one leaf
    if ( mlc.mesh_names.size() == 0 )
    {
        GGcerr << "MeshPhanLINACNav, no leaves in the mlc file were found!" << GGendl;
        exit_simulation();
    }

    // Check if the number of leaves match with the provided parameters
    if ( m_linac.A_nb_leaves + m_linac.B_nb_leaves !=  mlc.mesh_names.size() )
    {
        GGcerr << "MeshPhanLINACNav, number of leaves provided by the user is different to the number of meshes contained on the file!" << GGendl;
        exit_simulation();
    }

    // Some allocation
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.A_leaf_index), m_linac.A_nb_leaves * sizeof( ui32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.A_leaf_nb_triangles), m_linac.A_nb_leaves * sizeof( ui32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.A_leaf_aabb), m_linac.A_nb_leaves * sizeof( AabbData ) ) );

    HANDLE_ERROR( cudaMallocManaged( &(m_linac.B_leaf_index), m_linac.B_nb_leaves * sizeof( ui32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.B_leaf_nb_triangles), m_linac.B_nb_leaves * sizeof( ui32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.B_leaf_aabb), m_linac.B_nb_leaves * sizeof( AabbData ) ) );

//    GGcout << "first allocation" << GGendl;

    // Pre-calculation and checking of the data
    ui32 i_leaf = 0;
    std::string leaf_name, bank_name;
    ui32 index_leaf_bank;
    ui32 tot_tri_bank_A = 0;
    ui32 tot_tri_bank_B = 0;

    while ( i_leaf < mlc.mesh_names.size() )
    {
        // Get name of the leaf
        leaf_name = mlc.mesh_names[ i_leaf ];

        // Bank A or B
        bank_name = leaf_name[ 0 ];

        // Check
        if ( bank_name != "A" && bank_name != "B" )
        {
            GGcerr << "MeshPhanLINACNav: name of each leaf must start by the bank 'A' or 'B', " << bank_name << " given!" << GGendl;
            exit_simulation();
        }

        // Get leaf index
        index_leaf_bank = std::stoi( leaf_name.substr( 1, leaf_name.size()-1 ) );

        // If bank A
        if ( bank_name == "A" )
        {
            // Check
            if ( index_leaf_bank == 0 || index_leaf_bank > m_linac.A_nb_leaves )
            {
                GGcerr << "MeshPhanLINACNav: name of leaves must have index starting from 1 to N leaves!" << GGendl;
                exit_simulation();
            }

            // Store in sort way te number of triangles for each leaf
            // index_leaf_bank-1 because leaf start from 1 to N
            m_linac.A_leaf_nb_triangles[ index_leaf_bank-1 ] = mlc.nb_triangles[ i_leaf ];
            tot_tri_bank_A += mlc.nb_triangles[ i_leaf ];

//            GGcout << " A nb tri " << m_linac.A_leaf_nb_triangles[ index_leaf_bank-1 ] << " ileaf " << i_leaf << GGendl;

        }

        // If bank B
        if ( bank_name == "B" )
        {
            // Check
            if ( index_leaf_bank == 0 || index_leaf_bank > m_linac.B_nb_leaves )
            {
                GGcerr << "MeshPhanLINACNav: name of leaves must have index starting from 1 to N leaves!" << GGendl;
                exit_simulation();
            }

            // Store in sort way te number of triangles for each leaf
            // index_leaf_bank-1 because leaf start from 1 to N
            m_linac.B_leaf_nb_triangles[ index_leaf_bank-1 ] = mlc.nb_triangles[ i_leaf ];
            tot_tri_bank_B += mlc.nb_triangles[ i_leaf ];

//            GGcout << " B nb tri " << m_linac.B_leaf_nb_triangles[ index_leaf_bank-1 ] << " ileaf " << i_leaf << GGendl;
        }

        ++i_leaf;
    } // i_leaf

//    GGcout << "Check ok" << GGendl;

    // Compute the offset for each leaf from bank A
    m_linac.A_leaf_index[ 0 ] = 0;
    i_leaf = 1; while ( i_leaf < m_linac.A_nb_leaves )
    {
        m_linac.A_leaf_index[ i_leaf ] = m_linac.A_leaf_index[ i_leaf-1 ] + m_linac.A_leaf_nb_triangles[ i_leaf-1 ];

//        GGcout << " A offset " << m_linac.A_leaf_index[ i_leaf ]
//                  << " ileaf " << i_leaf << " nb tri: " << m_linac.A_leaf_nb_triangles[ i_leaf ] << GGendl;

        ++i_leaf;

    }

    // Compute the offset for each leaf from bank B
    m_linac.B_leaf_index[ 0 ] = 0;
    i_leaf = 1; while ( i_leaf < m_linac.B_nb_leaves )
    {
        m_linac.B_leaf_index[ i_leaf ] = m_linac.B_leaf_index[ i_leaf-1 ] + m_linac.B_leaf_nb_triangles[ i_leaf-1 ];
//        GGcout << " B offset " << m_linac.B_leaf_index[ i_leaf ]
//                  << " ileaf " << i_leaf << " nb tri: " << m_linac.B_leaf_nb_triangles[ i_leaf ] << GGendl;

        ++i_leaf;
    }

//    GGcout << "Get offset" << GGendl;

    // Some others allocations
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.A_leaf_v1), tot_tri_bank_A * sizeof( f32xyz ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.A_leaf_v2), tot_tri_bank_A * sizeof( f32xyz ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.A_leaf_v3), tot_tri_bank_A * sizeof( f32xyz ) ) );

    HANDLE_ERROR( cudaMallocManaged( &(m_linac.B_leaf_v1), tot_tri_bank_B * sizeof( f32xyz ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.B_leaf_v2), tot_tri_bank_B * sizeof( f32xyz ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.B_leaf_v3), tot_tri_bank_B * sizeof( f32xyz ) ) );

//    GGcout << "Second allocation" << GGendl;

    // Loop over leaf. Organize mesh data into the linac data.
    ui32 i_tri, offset_bank, offset_mlc;
    f32xyz v1, v2, v3;
    f32 xmin, xmax, ymin, ymax, zmin, zmax;
    i_leaf = 0; while ( i_leaf < mlc.mesh_names.size() )
    {
        // Get name of the leaf
        leaf_name = mlc.mesh_names[ i_leaf ];

        // Bank A or B
        bank_name = leaf_name[ 0 ];

        // Get leaf index within the bank
        index_leaf_bank = std::stoi( leaf_name.substr( 1, leaf_name.size()-1 ) ) - 1; // -1 because leaf start from 1 to N

        // index within the mlc (all meshes)
        offset_mlc = mlc.mesh_index[ i_leaf ];

//        GGcout << "leaf " << i_leaf << " name: " << leaf_name
//               << " bank: " << bank_name << " index: " << index_leaf_bank
//               << " offset: " << offset_mlc << GGendl;

        // Init AABB
        xmin = FLT_MAX; xmax = -FLT_MAX;
        ymin = FLT_MAX; ymax = -FLT_MAX;
        zmin = FLT_MAX; zmax = -FLT_MAX;

        // If bank A
        if ( bank_name == "A" )
        {
            // index within the bank
            offset_bank = m_linac.A_leaf_index[ index_leaf_bank ];

//            GGcout << "    A offset bank: " << offset_bank << GGendl;

//            GGcout << " Bank A leaft " << index_leaf_bank << GGendl;

            // loop over triangles
            i_tri = 0; while ( i_tri < m_linac.A_leaf_nb_triangles[ index_leaf_bank ] )
            {
                // Store on the right place
                v1 = mlc.v1[ offset_mlc + i_tri ];
                v2 = mlc.v2[ offset_mlc + i_tri ];
                v3 = mlc.v3[ offset_mlc + i_tri ];

//                printf("  v1 %f %f %f # v2 %f %f %f # v3 %f %f %f\n", v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, v3.x, v3.y, v3.z);

                m_linac.A_leaf_v1[ offset_bank + i_tri ] = v1;
                m_linac.A_leaf_v2[ offset_bank + i_tri ] = v2;
                m_linac.A_leaf_v3[ offset_bank + i_tri ] = v3;

                // Determine AABB
                if ( v1.x > xmax ) xmax = v1.x;
                if ( v2.x > xmax ) xmax = v2.x;
                if ( v3.x > xmax ) xmax = v3.x;

                if ( v1.y > ymax ) ymax = v1.y;
                if ( v2.y > ymax ) ymax = v2.y;
                if ( v3.y > ymax ) ymax = v3.y;

                if ( v1.z > zmax ) zmax = v1.z;
                if ( v2.z > zmax ) zmax = v2.z;
                if ( v3.z > zmax ) zmax = v3.z;

                if ( v1.x < xmin ) xmin = v1.x;
                if ( v2.x < xmin ) xmin = v2.x;
                if ( v3.x < xmin ) xmin = v3.x;

                if ( v1.y < ymin ) ymin = v1.y;
                if ( v2.y < ymin ) ymin = v2.y;
                if ( v3.y < ymin ) ymin = v3.y;

                if ( v1.z < zmin ) zmin = v1.z;
                if ( v2.z < zmin ) zmin = v2.z;
                if ( v3.z < zmin ) zmin = v3.z;

                ++i_tri;
            }

//            GGcout << "    A tri process" << GGendl;

            // Store the bounding box of the current leaf
            m_linac.A_leaf_aabb[ index_leaf_bank ].xmin = xmin;
            m_linac.A_leaf_aabb[ index_leaf_bank ].xmax = xmax;
            m_linac.A_leaf_aabb[ index_leaf_bank ].ymin = ymin;
            m_linac.A_leaf_aabb[ index_leaf_bank ].ymax = ymax;
            m_linac.A_leaf_aabb[ index_leaf_bank ].zmin = zmin;
            m_linac.A_leaf_aabb[ index_leaf_bank ].zmax = zmax;

//            GGcout << "    A aabb" << GGendl;
//            GGcout << " A" << index_leaf_bank << " aabb: " << xmin << " " << xmax << GGendl;

        }
        else // Bank B
        {
            // index within the bank
            offset_bank = m_linac.B_leaf_index[ index_leaf_bank ];

//            GGcout << "    B offset bank: " << offset_bank << GGendl;

            // loop over triangles
            i_tri = 0; while ( i_tri < m_linac.B_leaf_nb_triangles[ index_leaf_bank ] )
            {
                // Store on the right place
                v1 = mlc.v1[ offset_mlc + i_tri ];
                v2 = mlc.v2[ offset_mlc + i_tri ];
                v3 = mlc.v3[ offset_mlc + i_tri ];

                m_linac.B_leaf_v1[ offset_bank + i_tri ] = v1;
                m_linac.B_leaf_v2[ offset_bank + i_tri ] = v2;
                m_linac.B_leaf_v3[ offset_bank + i_tri ] = v3;

                // Determine AABB
                if ( v1.x > xmax ) xmax = v1.x;
                if ( v2.x > xmax ) xmax = v2.x;
                if ( v3.x > xmax ) xmax = v3.x;

                if ( v1.y > ymax ) ymax = v1.y;
                if ( v2.y > ymax ) ymax = v2.y;
                if ( v3.y > ymax ) ymax = v3.y;

                if ( v1.z > zmax ) zmax = v1.z;
                if ( v2.z > zmax ) zmax = v2.z;
                if ( v3.z > zmax ) zmax = v3.z;

                if ( v1.x < xmin ) xmin = v1.x;
                if ( v2.x < xmin ) xmin = v2.x;
                if ( v3.x < xmin ) xmin = v3.x;

                if ( v1.y < ymin ) ymin = v1.y;
                if ( v2.y < ymin ) ymin = v2.y;
                if ( v3.y < ymin ) ymin = v3.y;

                if ( v1.z < zmin ) zmin = v1.z;
                if ( v2.z < zmin ) zmin = v2.z;
                if ( v3.z < zmin ) zmin = v3.z;

                ++i_tri;
            }

//            GGcout << "    B tri process" << GGendl;

            // Store the bounding box of the current leaf
            m_linac.B_leaf_aabb[ index_leaf_bank ].xmin = xmin;
            m_linac.B_leaf_aabb[ index_leaf_bank ].xmax = xmax;
            m_linac.B_leaf_aabb[ index_leaf_bank ].ymin = ymin;
            m_linac.B_leaf_aabb[ index_leaf_bank ].ymax = ymax;
            m_linac.B_leaf_aabb[ index_leaf_bank ].zmin = zmin;
            m_linac.B_leaf_aabb[ index_leaf_bank ].zmax = zmax;

//            GGcout << "    B aabb " << GGendl;
//            GGcout << " B" << index_leaf_bank << " aabb: " << xmin << " " << xmax << GGendl;
        }

        ++i_leaf;
    } // i_leaf

//    GGcout << "Organize data" << GGendl;

    // Finally, compute the AABB of the bank A
    xmin = FLT_MAX; xmax = -FLT_MAX;
    ymin = FLT_MAX; ymax = -FLT_MAX;
    zmin = FLT_MAX; zmax = -FLT_MAX;
    i_leaf = 0; while ( i_leaf < m_linac.A_nb_leaves )
    {
        if ( m_linac.A_leaf_aabb[ i_leaf ].xmin < xmin ) xmin = m_linac.A_leaf_aabb[ i_leaf ].xmin;
        if ( m_linac.A_leaf_aabb[ i_leaf ].ymin < ymin ) ymin = m_linac.A_leaf_aabb[ i_leaf ].ymin;
        if ( m_linac.A_leaf_aabb[ i_leaf ].zmin < zmin ) zmin = m_linac.A_leaf_aabb[ i_leaf ].zmin;

        if ( m_linac.A_leaf_aabb[ i_leaf ].xmax > xmax ) xmax = m_linac.A_leaf_aabb[ i_leaf ].xmax;
        if ( m_linac.A_leaf_aabb[ i_leaf ].ymax > ymax ) ymax = m_linac.A_leaf_aabb[ i_leaf ].ymax;
        if ( m_linac.A_leaf_aabb[ i_leaf ].zmax > zmax ) zmax = m_linac.A_leaf_aabb[ i_leaf ].zmax;

        ++i_leaf;
    }

    m_linac.A_bank_aabb.xmin = xmin;
    m_linac.A_bank_aabb.xmax = xmax;
    m_linac.A_bank_aabb.ymin = ymin;
    m_linac.A_bank_aabb.ymax = ymax;
    m_linac.A_bank_aabb.zmin = zmin;
    m_linac.A_bank_aabb.zmax = zmax;

    // And for the bank B
    xmin = FLT_MAX; xmax = -FLT_MAX;
    ymin = FLT_MAX; ymax = -FLT_MAX;
    zmin = FLT_MAX; zmax = -FLT_MAX;
    i_leaf = 0; while ( i_leaf < m_linac.B_nb_leaves )
    {
        if ( m_linac.B_leaf_aabb[ i_leaf ].xmin < xmin ) xmin = m_linac.B_leaf_aabb[ i_leaf ].xmin;
        if ( m_linac.B_leaf_aabb[ i_leaf ].ymin < ymin ) ymin = m_linac.B_leaf_aabb[ i_leaf ].ymin;
        if ( m_linac.B_leaf_aabb[ i_leaf ].zmin < zmin ) zmin = m_linac.B_leaf_aabb[ i_leaf ].zmin;

        if ( m_linac.B_leaf_aabb[ i_leaf ].xmax > xmax ) xmax = m_linac.B_leaf_aabb[ i_leaf ].xmax;
        if ( m_linac.B_leaf_aabb[ i_leaf ].ymax > ymax ) ymax = m_linac.B_leaf_aabb[ i_leaf ].ymax;
        if ( m_linac.B_leaf_aabb[ i_leaf ].zmax > zmax ) zmax = m_linac.B_leaf_aabb[ i_leaf ].zmax;

        ++i_leaf;
    }

    m_linac.B_bank_aabb.xmin = xmin;
    m_linac.B_bank_aabb.xmax = xmax;
    m_linac.B_bank_aabb.ymin = ymin;
    m_linac.B_bank_aabb.ymax = ymax;
    m_linac.B_bank_aabb.zmin = zmin;
    m_linac.B_bank_aabb.zmax = zmax;

//    GGcout << "Get AABB" << GGendl;

}


void MeshPhanLINACNav::m_init_jaw_x()
{
    // First check the file
    std::string ext = m_jaw_x_filename.substr( m_jaw_x_filename.find_last_of( "." ) + 1 );
    if ( ext != "obj" )
    {
        GGcerr << "MeshPhanLINACNav can only read mesh data in Wavefront format (.obj)!" << GGendl;
        exit_simulation();
    }

    // Then get data
    MeshIO *meshio = new MeshIO;
    MeshData jaw = meshio->read_mesh_file( m_jaw_x_filename );

    // Check if there are at least one jaw
    if ( jaw.mesh_names.size() == 0 )
    {
        GGcerr << "MeshPhanLINACNav, no jaw in the x-jaw file were found!" << GGendl;
        exit_simulation();
    }

    m_linac.X_nb_jaw = jaw.mesh_names.size();

    // Some allocation
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.X_jaw_index), m_linac.X_nb_jaw * sizeof( ui32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.X_jaw_nb_triangles), m_linac.X_nb_jaw * sizeof( ui32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.X_jaw_aabb), m_linac.X_nb_jaw * sizeof( AabbData ) ) );

    // Pre-calculation and checking of the data
    ui32 i_jaw = 0;
    std::string jaw_name, axis_name;
    ui32 index_jaw;
    ui32 tot_tri_jaw = 0;

    while ( i_jaw < m_linac.X_nb_jaw )
    {
        // Get name of the jaw
        jaw_name = jaw.mesh_names[ i_jaw ];

        // Name axis
        axis_name = jaw_name[ 0 ];

        // Check
        if ( axis_name != "X" )
        {
            GGcerr << "MeshPhanLINACNav: name of each jaw (in X) must start by 'X', " << axis_name << " given!" << GGendl;
            exit_simulation();
        }

        // Get leaf index
        index_jaw = std::stoi( jaw_name.substr( 1, jaw_name.size()-1 ) );

        // Check
        if ( index_jaw == 0 || index_jaw > 2 )
        {
            GGcerr << "MeshPhanLINACNav: name of jaws must have index starting from 1 to 2!" << GGendl;
            exit_simulation();
        }

        // Store the number of triangles for each jaw
        // index-1 because jaw start from 1 to 2
        m_linac.X_jaw_nb_triangles[ index_jaw-1 ] = jaw.nb_triangles[ i_jaw ];
        tot_tri_jaw += jaw.nb_triangles[ i_jaw ];

        ++i_jaw;
    } // i_leaf

    // Compute the offset for each jaw
    m_linac.X_jaw_index[ 0 ] = 0;
    m_linac.X_jaw_index[ 1 ] = m_linac.X_jaw_nb_triangles[ 0 ];

    // Some others allocations
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.X_jaw_v1), tot_tri_jaw * sizeof( f32xyz ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.X_jaw_v2), tot_tri_jaw * sizeof( f32xyz ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.X_jaw_v3), tot_tri_jaw * sizeof( f32xyz ) ) );

    // Loop over leaf. Organize mesh data into the linac data.
    ui32 i_tri, offset_mesh, offset_linac;
    f32xyz v1, v2, v3;
    f32 xmin, xmax, ymin, ymax, zmin, zmax;
    i_jaw = 0; while ( i_jaw < m_linac.X_nb_jaw )
    {
        // Get name of the leaf
        jaw_name = jaw.mesh_names[ i_jaw ];

        // Get leaf index within the bank
        index_jaw = std::stoi( jaw_name.substr( 1, jaw_name.size()-1 ) ) - 1; // -1 because jaw start from 1 to 2

        // index within the mlc (all meshes)
        offset_mesh = jaw.mesh_index[ i_jaw ];

        // Init AABB
        xmin = FLT_MAX; xmax = -FLT_MAX;
        ymin = FLT_MAX; ymax = -FLT_MAX;
        zmin = FLT_MAX; zmax = -FLT_MAX;

        // index within the bank
        offset_linac = m_linac.X_jaw_index[ index_jaw ];

        // loop over triangles
        i_tri = 0; while ( i_tri < m_linac.X_jaw_nb_triangles[ index_jaw ] )
        {
            // Store on the right place
            v1 = jaw.v1[ offset_mesh + i_tri ];
            v2 = jaw.v2[ offset_mesh + i_tri ];
            v3 = jaw.v3[ offset_mesh + i_tri ];

            m_linac.X_jaw_v1[ offset_linac + i_tri ] = v1;
            m_linac.X_jaw_v2[ offset_linac + i_tri ] = v2;
            m_linac.X_jaw_v3[ offset_linac + i_tri ] = v3;

            // Determine AABB
            if ( v1.x > xmax ) xmax = v1.x;
            if ( v2.x > xmax ) xmax = v2.x;
            if ( v3.x > xmax ) xmax = v3.x;

            if ( v1.y > ymax ) ymax = v1.y;
            if ( v2.y > ymax ) ymax = v2.y;
            if ( v3.y > ymax ) ymax = v3.y;

            if ( v1.z > zmax ) zmax = v1.z;
            if ( v2.z > zmax ) zmax = v2.z;
            if ( v3.z > zmax ) zmax = v3.z;

            if ( v1.x < xmin ) xmin = v1.x;
            if ( v2.x < xmin ) xmin = v2.x;
            if ( v3.x < xmin ) xmin = v3.x;

            if ( v1.y < ymin ) ymin = v1.y;
            if ( v2.y < ymin ) ymin = v2.y;
            if ( v3.y < ymin ) ymin = v3.y;

            if ( v1.z < zmin ) zmin = v1.z;
            if ( v2.z < zmin ) zmin = v2.z;
            if ( v3.z < zmin ) zmin = v3.z;

            ++i_tri;
        }

        // Store the bounding box of the current jaw
        m_linac.X_jaw_aabb[ index_jaw ].xmin = xmin;
        m_linac.X_jaw_aabb[ index_jaw ].xmax = xmax;
        m_linac.X_jaw_aabb[ index_jaw ].ymin = ymin;
        m_linac.X_jaw_aabb[ index_jaw ].ymax = ymax;
        m_linac.X_jaw_aabb[ index_jaw ].zmin = zmin;
        m_linac.X_jaw_aabb[ index_jaw ].zmax = zmax;

        ++i_jaw;
    } // i_jaw

}

void MeshPhanLINACNav::m_init_jaw_y()
{
    // First check the file
    std::string ext = m_jaw_y_filename.substr( m_jaw_y_filename.find_last_of( "." ) + 1 );
    if ( ext != "obj" )
    {
        GGcerr << "MeshPhanLINACNav can only read mesh data in Wavefront format (.obj)!" << GGendl;
        exit_simulation();
    }

    // Then get data
    MeshIO *meshio = new MeshIO;
    MeshData jaw = meshio->read_mesh_file( m_jaw_y_filename );

    // Check if there are at least one jaw
    if ( jaw.mesh_names.size() == 0 )
    {
        GGcerr << "MeshPhanLINACNav, no jaw in the y-jaw file were found!" << GGendl;
        exit_simulation();
    }

    m_linac.Y_nb_jaw = jaw.mesh_names.size();

    // Some allocation
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.Y_jaw_index), m_linac.Y_nb_jaw * sizeof( ui32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.Y_jaw_nb_triangles), m_linac.Y_nb_jaw * sizeof( ui32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.Y_jaw_aabb), m_linac.Y_nb_jaw * sizeof( AabbData ) ) );

    // Pre-calculation and checking of the data
    ui32 i_jaw = 0;
    std::string jaw_name, axis_name;
    ui32 index_jaw;
    ui32 tot_tri_jaw = 0;

    while ( i_jaw < m_linac.Y_nb_jaw )
    {
        // Get name of the jaw
        jaw_name = jaw.mesh_names[ i_jaw ];

        // Name axis
        axis_name = jaw_name[ 0 ];

        // Check
        if ( axis_name != "Y" )
        {
            GGcerr << "MeshPhanLINACNav: name of each jaw (in Y) must start by 'Y', " << axis_name << " given!" << GGendl;
            exit_simulation();
        }

        // Get leaf index
        index_jaw = std::stoi( jaw_name.substr( 1, jaw_name.size()-1 ) );

        // Check
        if ( index_jaw == 0 || index_jaw > 2 )
        {
            GGcerr << "MeshPhanLINACNav: name of jaws must have index starting from 1 to 2!" << GGendl;
            exit_simulation();
        }

        // Store the number of triangles for each jaw
        // index-1 because jaw start from 1 to 2
        m_linac.Y_jaw_nb_triangles[ index_jaw-1 ] = jaw.nb_triangles[ i_jaw ];
        tot_tri_jaw += jaw.nb_triangles[ i_jaw ];

        ++i_jaw;
    } // i_leaf

    // Compute the offset for each jaw
    m_linac.Y_jaw_index[ 0 ] = 0;
    m_linac.Y_jaw_index[ 1 ] = m_linac.Y_jaw_nb_triangles[ 0 ];

    // Some others allocations
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.Y_jaw_v1), tot_tri_jaw * sizeof( f32xyz ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.Y_jaw_v2), tot_tri_jaw * sizeof( f32xyz ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac.Y_jaw_v3), tot_tri_jaw * sizeof( f32xyz ) ) );

    // Loop over leaf. Organize mesh data into the linac data.
    ui32 i_tri, offset_mesh, offset_linac;
    f32xyz v1, v2, v3;
    f32 xmin, xmax, ymin, ymax, zmin, zmax;
    i_jaw = 0; while ( i_jaw < m_linac.Y_nb_jaw )
    {
        // Get name of the leaf
        jaw_name = jaw.mesh_names[ i_jaw ];

        // Get leaf index within the bank
        index_jaw = std::stoi( jaw_name.substr( 1, jaw_name.size()-1 ) ) - 1; // -1 because jaw start from 1 to 2

        // index within the mlc (all meshes)
        offset_mesh = jaw.mesh_index[ i_jaw ];

        // Init AABB
        xmin = FLT_MAX; xmax = -FLT_MAX;
        ymin = FLT_MAX; ymax = -FLT_MAX;
        zmin = FLT_MAX; zmax = -FLT_MAX;

        // index within the bank
        offset_linac = m_linac.Y_jaw_index[ index_jaw ];

        // loop over triangles
        i_tri = 0; while ( i_tri < m_linac.Y_jaw_nb_triangles[ index_jaw ] )
        {
            // Store on the right place
            v1 = jaw.v1[ offset_mesh + i_tri ];
            v2 = jaw.v2[ offset_mesh + i_tri ];
            v3 = jaw.v3[ offset_mesh + i_tri ];

            m_linac.Y_jaw_v1[ offset_linac + i_tri ] = v1;
            m_linac.Y_jaw_v2[ offset_linac + i_tri ] = v2;
            m_linac.Y_jaw_v3[ offset_linac + i_tri ] = v3;

            // Determine AABB
            if ( v1.x > xmax ) xmax = v1.x;
            if ( v2.x > xmax ) xmax = v2.x;
            if ( v3.x > xmax ) xmax = v3.x;

            if ( v1.y > ymax ) ymax = v1.y;
            if ( v2.y > ymax ) ymax = v2.y;
            if ( v3.y > ymax ) ymax = v3.y;

            if ( v1.z > zmax ) zmax = v1.z;
            if ( v2.z > zmax ) zmax = v2.z;
            if ( v3.z > zmax ) zmax = v3.z;

            if ( v1.x < xmin ) xmin = v1.x;
            if ( v2.x < xmin ) xmin = v2.x;
            if ( v3.x < xmin ) xmin = v3.x;

            if ( v1.y < ymin ) ymin = v1.y;
            if ( v2.y < ymin ) ymin = v2.y;
            if ( v3.y < ymin ) ymin = v3.y;

            if ( v1.z < zmin ) zmin = v1.z;
            if ( v2.z < zmin ) zmin = v2.z;
            if ( v3.z < zmin ) zmin = v3.z;

            ++i_tri;
        }

        // Store the bounding box of the current jaw
        m_linac.Y_jaw_aabb[ index_jaw ].xmin = xmin;
        m_linac.Y_jaw_aabb[ index_jaw ].xmax = xmax;
        m_linac.Y_jaw_aabb[ index_jaw ].ymin = ymin;
        m_linac.Y_jaw_aabb[ index_jaw ].ymax = ymax;
        m_linac.Y_jaw_aabb[ index_jaw ].zmin = zmin;
        m_linac.Y_jaw_aabb[ index_jaw ].zmax = zmax;

        ++i_jaw;
    } // i_jaw

}

void MeshPhanLINACNav::m_translate_jaw_x( ui32 index, f32xyz T )
{
    ui32 offset = m_linac.X_jaw_index[ index ];
    ui32 nb_tri = m_linac.X_jaw_nb_triangles[ index ];

    ui32 i_tri = 0; while ( i_tri < nb_tri )
    {
        m_linac.X_jaw_v1[ offset + i_tri ] = fxyz_add( m_linac.X_jaw_v1[ offset + i_tri ], T );
        m_linac.X_jaw_v2[ offset + i_tri ] = fxyz_add( m_linac.X_jaw_v2[ offset + i_tri ], T );
        m_linac.X_jaw_v3[ offset + i_tri ] = fxyz_add( m_linac.X_jaw_v3[ offset + i_tri ], T );
        ++i_tri;
    }

    // Move as well the AABB
    m_linac.X_jaw_aabb[ index ].xmin += T.x;
    m_linac.X_jaw_aabb[ index ].xmax += T.x;
    m_linac.X_jaw_aabb[ index ].ymin += T.y;
    m_linac.X_jaw_aabb[ index ].ymax += T.y;
    m_linac.X_jaw_aabb[ index ].zmin += T.z;
    m_linac.X_jaw_aabb[ index ].zmax += T.z;
}

void MeshPhanLINACNav::m_translate_jaw_y( ui32 index, f32xyz T )
{
    ui32 offset = m_linac.Y_jaw_index[ index ];
    ui32 nb_tri = m_linac.Y_jaw_nb_triangles[ index ];

    ui32 i_tri = 0; while ( i_tri < nb_tri )
    {
        m_linac.Y_jaw_v1[ offset + i_tri ] = fxyz_add( m_linac.Y_jaw_v1[ offset + i_tri ], T );
        m_linac.Y_jaw_v2[ offset + i_tri ] = fxyz_add( m_linac.Y_jaw_v2[ offset + i_tri ], T );
        m_linac.Y_jaw_v3[ offset + i_tri ] = fxyz_add( m_linac.Y_jaw_v3[ offset + i_tri ], T );
        ++i_tri;
    }

    // Move as well the AABB
    m_linac.Y_jaw_aabb[ index ].xmin += T.x;
    m_linac.Y_jaw_aabb[ index ].xmax += T.x;
    m_linac.Y_jaw_aabb[ index ].ymin += T.y;
    m_linac.Y_jaw_aabb[ index ].ymax += T.y;
    m_linac.Y_jaw_aabb[ index ].zmin += T.z;
    m_linac.Y_jaw_aabb[ index ].zmax += T.z;
}

void MeshPhanLINACNav::m_translate_leaf_A( ui32 index, f32xyz T )
{
    ui32 offset = m_linac.A_leaf_index[ index ];
    ui32 nb_tri = m_linac.A_leaf_nb_triangles[ index ];

    ui32 i_tri = 0; while ( i_tri < nb_tri )
    {
        m_linac.A_leaf_v1[ offset + i_tri ] = fxyz_add( m_linac.A_leaf_v1[ offset + i_tri ], T );
        m_linac.A_leaf_v2[ offset + i_tri ] = fxyz_add( m_linac.A_leaf_v2[ offset + i_tri ], T );
        m_linac.A_leaf_v3[ offset + i_tri ] = fxyz_add( m_linac.A_leaf_v3[ offset + i_tri ], T );
        ++i_tri;
    }

    // Move as well the AABB
    m_linac.A_leaf_aabb[ index ].xmin += T.x;
    m_linac.A_leaf_aabb[ index ].xmax += T.x;
    m_linac.A_leaf_aabb[ index ].ymin += T.y;
    m_linac.A_leaf_aabb[ index ].ymax += T.y;
    m_linac.A_leaf_aabb[ index ].zmin += T.z;
    m_linac.A_leaf_aabb[ index ].zmax += T.z;

    // Update the bank AABB
    if ( m_linac.A_leaf_aabb[ index ].xmin < m_linac.A_bank_aabb.xmin )
    {
        m_linac.A_bank_aabb.xmin = m_linac.A_leaf_aabb[ index ].xmin;
    }

    if ( m_linac.A_leaf_aabb[ index ].ymin < m_linac.A_bank_aabb.ymin )
    {
        m_linac.A_bank_aabb.ymin = m_linac.A_leaf_aabb[ index ].ymin;
    }

    if ( m_linac.A_leaf_aabb[ index ].zmin < m_linac.A_bank_aabb.zmin )
    {
        m_linac.A_bank_aabb.zmin = m_linac.A_leaf_aabb[ index ].zmin;
    }

    if ( m_linac.A_leaf_aabb[ index ].xmax > m_linac.A_bank_aabb.xmax )
    {
        m_linac.A_bank_aabb.xmax = m_linac.A_leaf_aabb[ index ].xmax;
    }

    if ( m_linac.A_leaf_aabb[ index ].ymax > m_linac.A_bank_aabb.ymax )
    {
        m_linac.A_bank_aabb.ymax = m_linac.A_leaf_aabb[ index ].ymax;
    }

    if ( m_linac.A_leaf_aabb[ index ].zmax > m_linac.A_bank_aabb.zmax )
    {
        m_linac.A_bank_aabb.zmax = m_linac.A_leaf_aabb[ index ].zmax;
    }

}

void MeshPhanLINACNav::m_translate_leaf_B( ui32 index, f32xyz T )
{
    ui32 offset = m_linac.B_leaf_index[ index ];
    ui32 nb_tri = m_linac.B_leaf_nb_triangles[ index ];

    ui32 i_tri = 0; while ( i_tri < nb_tri )
    {
        m_linac.B_leaf_v1[ offset + i_tri ] = fxyz_add( m_linac.B_leaf_v1[ offset + i_tri ], T );
        m_linac.B_leaf_v2[ offset + i_tri ] = fxyz_add( m_linac.B_leaf_v2[ offset + i_tri ], T );
        m_linac.B_leaf_v3[ offset + i_tri ] = fxyz_add( m_linac.B_leaf_v3[ offset + i_tri ], T );
        ++i_tri;
    }

    // Move as well the AABB
    m_linac.B_leaf_aabb[ index ].xmin += T.x;
    m_linac.B_leaf_aabb[ index ].xmax += T.x;
    m_linac.B_leaf_aabb[ index ].ymin += T.y;
    m_linac.B_leaf_aabb[ index ].ymax += T.y;
    m_linac.B_leaf_aabb[ index ].zmin += T.z;
    m_linac.B_leaf_aabb[ index ].zmax += T.z;

    // Update the bank AABB
    if ( m_linac.B_leaf_aabb[ index ].xmin < m_linac.B_bank_aabb.xmin )
    {
        m_linac.B_bank_aabb.xmin = m_linac.B_leaf_aabb[ index ].xmin;
    }

    if ( m_linac.B_leaf_aabb[ index ].ymin < m_linac.B_bank_aabb.ymin )
    {
        m_linac.B_bank_aabb.ymin = m_linac.B_leaf_aabb[ index ].ymin;
    }

    if ( m_linac.B_leaf_aabb[ index ].zmin < m_linac.B_bank_aabb.zmin )
    {
        m_linac.B_bank_aabb.zmin = m_linac.B_leaf_aabb[ index ].zmin;
    }

    if ( m_linac.B_leaf_aabb[ index ].xmax > m_linac.B_bank_aabb.xmax )
    {
        m_linac.B_bank_aabb.xmax = m_linac.B_leaf_aabb[ index ].xmax;
    }

    if ( m_linac.B_leaf_aabb[ index ].ymax > m_linac.B_bank_aabb.ymax )
    {
        m_linac.B_bank_aabb.ymax = m_linac.B_leaf_aabb[ index ].ymax;
    }

    if ( m_linac.B_leaf_aabb[ index ].zmax > m_linac.B_bank_aabb.zmax )
    {
        m_linac.B_bank_aabb.zmax = m_linac.B_leaf_aabb[ index ].zmax;
    }

}

void MeshPhanLINACNav::m_configure_linac()
{

    // Open the beam file
    std::ifstream file( m_beam_config_filename.c_str(), std::ios::in );
    if( !file )
    {
        GGcerr << "Error to open the Beam file'" << m_beam_config_filename << "'!" << GGendl;
        exit_simulation();
    }

    std::string line;
    std::vector< std::string > keys;

    // Look for the beam number
    while ( file )
    {
        // Read a line
        std::getline( file, line );
        keys = m_split_txt( line );

        if ( keys.size() >= 3 )
        {
            if ( keys[ 0 ] == "Beam" && std::stoi( keys[ 2 ] ) == m_beam_index )
            {
                break;
            }
        }
    }

    GGcout << "Find beam: " << line << GGendl;

    // Then look for the number of fields
    while ( file )
    {
        // Read a line
        std::getline( file, line );

        if ( line.find("Number of Fields") != std::string::npos )
        {
            break;
        }
    }

    GGcout << "Find nb field: " << line << GGendl;

    keys = m_split_txt( line );
    ui32 nb_fields = std::stoi( keys[ 4 ] );

    if ( m_field_index >= nb_fields )
    {
        GGcerr << "Out of index for the field number, asked: " << m_field_index
               << " but a total of field of " << nb_fields << GGendl;
        exit_simulation();
    }    

    // Look for the number of leaves
    while ( file )
    {
        // Read a line
        std::getline( file, line );

        if ( line.find("Number of Leaves") != std::string::npos )
        {
            break;
        }
    }

    GGcout << "Find nb leaves: " << line << GGendl;

    keys = m_split_txt( line );
    ui32 nb_leaves = std::stoi( keys[ 4 ] );
    if ( m_linac.A_nb_leaves + m_linac.B_nb_leaves != nb_leaves )
    {
        GGcerr << "Beam configuration error, " << nb_leaves
               << " leaves were found but LINAC model have " << m_linac.A_nb_leaves + m_linac.B_nb_leaves
               << " leaves!" << GGendl;
        exit_simulation();
    }

    // Search the required field
    while ( file )
    {
        // Read a line
        std::getline( file, line );
        keys = m_split_txt( line );

        if ( keys.size() >= 3 )
        {
            if ( keys[ 0 ] == "Control" && std::stoi( keys[ 2 ] ) == m_field_index )
            {
                break;
            }
        }
    }

    GGcout << "Find field: " << line << GGendl;

    // Then read the index CDF (not use at the time, so skip the line)
    std::getline( file, line );

    // Get the gantry angle
    std::getline( file, line );

    // Check
    if ( line.find( "Gantry Angle" ) == std::string::npos )
    {
        GGcerr << "Beam configuration error, no gantry angle was found!" << GGendl;
        exit_simulation();
    }

    // Read gantry angle values
    keys = m_split_txt( line );

    // if only one angle, rotate around the z-axis
    if ( keys.size() == 4 )
    {
        m_rot_linac = make_f32xyz( 0.0, 0.0, std::stof( keys[ 3 ] ) *deg );
    }
    else if ( keys.size() == 6 ) // non-coplanar beam, or rotation on the carousel
    {
        m_rot_linac = make_f32xyz( std::stof( keys[ 3 ] ) *deg,
                                   std::stof( keys[ 4 ] ) *deg,
                                   std::stof( keys[ 5 ] ) *deg );
    }
    else // otherwise, it seems that there is an error somewhere
    {
        GGcerr << "Beam configuration error, gantry angle must have one angle or the three rotation angles: "
               << keys.size() - 3 << " angles found!" << GGendl;
        exit_simulation();
    }

    // Get the transformation matrix to map local to global coordinate
    TransformCalculator *trans = new TransformCalculator;
    trans->set_translation( m_pos_mlc );
    trans->set_rotation( m_rot_linac );
    trans->set_axis_transformation( m_axis_linac );
    m_transform_linac = trans->get_transformation_matrix();
    delete trans;

    //// JAWS //////////////////////////////////////////

    // Next four lines should the jaw config
    f32 jaw_x_min = 0.0; bool jaw_x = false;
    f32 jaw_x_max = 0.0;
    f32 jaw_y_min = 0.0; bool jaw_y = false;
    f32 jaw_y_max = 0.0;

    while ( file )
    {
        // Read a line
        std::getline( file, line );

        if ( line.find( "Jaw" ) != std::string::npos )
        {
            keys = m_split_txt( line );
            if ( keys[ 1 ] == "X" && keys[ 2 ] == "min" )
            {
                jaw_x_min = std::stof( keys[ 4 ] );
                jaw_x = true;
            }
            if ( keys[ 1 ] == "X" && keys[ 2 ] == "max" )
            {
                jaw_x_max = std::stof( keys[ 4 ] );
                jaw_x = true;
            }
            if ( keys[ 1 ] == "Y" && keys[ 2 ] == "min" )
            {
                jaw_y_min = std::stof( keys[ 4 ] );
                jaw_y = true;
            }
            if ( keys[ 1 ] == "Y" && keys[ 2 ] == "max" )
            {
                jaw_y_max = std::stof( keys[ 4 ] );
                jaw_y = true;
            }
        }
        else
        {
            break;
        }
    }

    // Check
    if ( !jaw_x && m_linac.X_nb_jaw != 0 )
    {
        GGcerr << "Beam configuration error, geometry of the jaw-X was defined but the position values were not found!" << GGendl;
        exit_simulation();
    }
    if ( !jaw_y && m_linac.Y_nb_jaw != 0 )
    {
        GGcerr << "Beam configuration error, geometry of the jaw-Y was defined but the position values were not found!" << GGendl;
        exit_simulation();
    }

    // Configure the jaws
    if ( m_linac.X_nb_jaw != 0 )
    {
        m_translate_jaw_x( 0, make_f32xyz( jaw_x_max, 0.0, 0.0 ) );   // X1 ( x > 0 )
        m_translate_jaw_x( 1, make_f32xyz( jaw_x_min, 0.0, 0.0 ) );   // X2 ( x < 0 )
    }

    if ( m_linac.Y_nb_jaw != 0 )
    {
        m_translate_jaw_y( 0, make_f32xyz( 0.0, jaw_y_max, 0.0 ) );   // Y1 ( y > 0 )
        m_translate_jaw_y( 1, make_f32xyz( 0.0, jaw_y_min, 0.0 ) );   // Y2 ( y < 0 )
    }

    //// LEAVES BANK A ///////////////////////////////////////////////

    ui32 ileaf = 0;
    bool wd_leaf = false; // watchdog
    while ( file )
    {
        if ( line.find( "Leaf" ) != std::string::npos && line.find( "A" ) != std::string::npos )
        {
            // If first leaf of the bank A, check
            if ( ileaf == 0 )
            {
                keys = m_split_txt( line );
                if ( keys[ 1 ] != "1A" )
                {
                    GGcerr << "Beam configuration error, first leaf of the bank A must start by index '1A': " << keys[ 1 ]
                           << " found." << GGendl;
                    exit_simulation();
                }
            }

            // watchdog
            if ( ileaf >= m_linac.A_nb_leaves )
            {
                GGcerr << "Beam configuration error, find more leaves in the configuration "
                       << "file for the bank A than leaves in the LINAC model!" << GGendl;
                exit_simulation();
            }

            // find at least one leaf
            if ( !wd_leaf ) wd_leaf = true;

            // read data and move the leaf
            keys = m_split_txt( line );
            m_translate_leaf_A( ileaf++, make_f32xyz( std::stof( keys[ 3 ] ), 0.0, 0.0 ) );

        }
        else
        {
            break;
        }

        // Read a line
        std::getline( file, line );
    }

    // No leaves were found
    if ( !wd_leaf )
    {
        GGcerr << "Beam configuration error, no leaves from the bank A were found!" << GGendl;
        exit_simulation();
    }

    //// LEAVES BANK B ///////////////////////////////////////////////

    ileaf = 0;
    wd_leaf = false; // watchdog
    while ( file )
    {

        if ( line.find( "Leaf" ) != std::string::npos && line.find( "B" ) != std::string::npos )
        {
            // If first leaf of the bank A, check
            if ( ileaf == 0 )
            {
                keys = m_split_txt( line );
                if ( keys[ 1 ] != "1B" )
                {
                    GGcerr << "Beam configuration error, first leaf of the bank B must start by index '1B': " << keys[ 1 ]
                           << " found." << GGendl;
                    exit_simulation();
                }
            }

            // watchdog
            if ( ileaf >= m_linac.B_nb_leaves )
            {
                GGcerr << "Beam configuration error, find more leaves in the configuration "
                       << "file for the bank B than leaves in the LINAC model!" << GGendl;
                exit_simulation();
            }

            // find at least one leaf
            if ( !wd_leaf ) wd_leaf = true;

            // read data and move the leaf
            keys = m_split_txt( line );
            m_translate_leaf_B( ileaf++, make_f32xyz( std::stof( keys[ 3 ] ), 0.0, 0.0 ) );

        }
        else
        {
            break;
        }

        // Read a line
        std::getline( file, line );
    }

    // No leaves were found
    if ( !wd_leaf )
    {
        GGcerr << "Beam configuration error, no leaves from the bank B were found!" << GGendl;
        exit_simulation();
    }


}

// return memory usage
ui64 MeshPhanLINACNav::m_get_memory_usage()
{
    /*
    ui64 mem = 0;

    // First the voxelized phantom
    mem += ( m_phantom.data_h.number_of_voxels * sizeof( ui16 ) );
    // Then material data
    mem += ( ( 3 * m_materials.data_h.nb_elements_total + 23 * m_materials.data_h.nb_materials ) * sizeof( f32 ) );
    // Then cross sections (gamma)
    ui64 n = m_cross_sections.photon_CS.data_h.nb_bins;
    ui64 k = m_cross_sections.photon_CS.data_h.nb_mat;
    mem += ( ( n + 3*n*k + 3*101*n ) * sizeof( f32 ) );
    // Cross section (electron)
    mem += ( n*k*7*sizeof( f32 ) );
    // Finally the dose map
    n = m_dose_calculator.dose.tot_nb_dosels;
    mem += ( 2*n*sizeof( f64 ) + n*sizeof( ui32 ) );
    mem += ( 20 * sizeof( f32 ) );

    // If TLE
    if ( m_flag_TLE )
    {
        n = m_mu_table.nb_bins;
        mem += ( n*k*2 * sizeof( f32 ) ); // mu and mu_en
        mem += ( n*sizeof( f32 ) );       // energies
    }

    // If seTLE
    if ( m_flag_TLE == seTLE )
    {
        mem += ( m_phantom.data_h.number_of_voxels * ( sizeof( ui32 ) + sizeof( f32 ) ) );
    }

    return mem;
    */
}

//// Setting/Getting functions

void MeshPhanLINACNav::set_mlc_meshes( std::string filename )
{
    m_mlc_filename = filename;
}

void MeshPhanLINACNav::set_jaw_x_meshes( std::string filename )
{
    m_jaw_x_filename = filename;
}

void MeshPhanLINACNav::set_jaw_y_meshes( std::string filename )
{
    m_jaw_y_filename = filename;
}

void MeshPhanLINACNav::set_beam_configuration( std::string filename, ui32 beam_index, ui32 field_index )
{
    m_beam_config_filename = filename;
    m_beam_index = beam_index;
    m_field_index = field_index;
}

void MeshPhanLINACNav::set_number_of_leaves( ui32 nb_bank_A, ui32 nb_bank_B )
{
    m_linac.A_nb_leaves = nb_bank_A;
    m_linac.B_nb_leaves = nb_bank_B;
}

void MeshPhanLINACNav::set_mlc_position( f32 px, f32 py, f32 pz )
{
    m_pos_mlc = make_f32xyz( px, py, pz );
}

void MeshPhanLINACNav::set_local_jaw_x_position( f32 px, f32 py, f32 pz )
{
    m_loc_pos_jaw_x = make_f32xyz( px, py, pz );
}

void MeshPhanLINACNav::set_local_jaw_y_position( f32 px, f32 py, f32 pz )
{
    m_loc_pos_jaw_y = make_f32xyz( px, py, pz );
}


void MeshPhanLINACNav::set_linac_local_axis( f32 m00, f32 m01, f32 m02,
                                             f32 m10, f32 m11, f32 m12,
                                             f32 m20, f32 m21, f32 m22 )
{
    m_axis_linac = make_f32matrix33( m00, m01, m02,
                                     m10, m11, m12,
                                     m20, m21, m22 );
}



LinacData MeshPhanLINACNav::get_linac_geometry()
{
    return m_linac;
}

f32matrix44 MeshPhanLINACNav::get_linac_transformation()
{
    return m_transform_linac;
}

//void MeshPhanLINACNav::set_materials(std::string filename )
//{
//    m_materials_filename = filename;
//}

////// Main functions

MeshPhanLINACNav::MeshPhanLINACNav ()
{
    // Leaves in Bank A
    m_linac.A_leaf_v1 = NULL;           // Vertex 1  - Triangular meshes
    m_linac.A_leaf_v2 = NULL;           // Vertex 2
    m_linac.A_leaf_v3 = NULL;           // Vertex 3
    m_linac.A_leaf_index = NULL;        // Index to acces to a leaf
    m_linac.A_leaf_nb_triangles = NULL; // Nb of triangles within each leaf
    m_linac.A_leaf_aabb = NULL;         // Bounding box of each leaf

    m_linac.A_bank_aabb.xmin = 0.0;     // Bounding box of the bank A
    m_linac.A_bank_aabb.xmax = 0.0;
    m_linac.A_bank_aabb.ymin = 0.0;
    m_linac.A_bank_aabb.ymax = 0.0;
    m_linac.A_bank_aabb.zmin = 0.0;
    m_linac.A_bank_aabb.zmax = 0.0;

    m_linac.A_nb_leaves = 0;            // Number of leaves in the bank A

    // Leaves in Bank B
    m_linac.B_leaf_v1 = NULL;           // Vertex 1  - Triangular meshes
    m_linac.B_leaf_v2 = NULL;           // Vertex 2
    m_linac.B_leaf_v3 = NULL;           // Vertex 3
    m_linac.B_leaf_index = NULL;        // Index to acces to a leaf
    m_linac.B_leaf_nb_triangles = NULL; // Nb of triangles within each leaf
    m_linac.B_leaf_aabb = NULL;         // Bounding box of each leaf

    m_linac.B_bank_aabb.xmin = 0.0;     // Bounding box of the bank B
    m_linac.B_bank_aabb.xmax = 0.0;
    m_linac.B_bank_aabb.ymin = 0.0;
    m_linac.B_bank_aabb.ymax = 0.0;
    m_linac.B_bank_aabb.zmin = 0.0;
    m_linac.B_bank_aabb.zmax = 0.0;

    m_linac.B_nb_leaves = 0;            // Number of leaves in the bank B

    // Jaws X
    m_linac.X_jaw_v1 = NULL;           // Vertex 1  - Triangular meshes
    m_linac.X_jaw_v2 = NULL;           // Vertex 2
    m_linac.X_jaw_v3 = NULL;           // Vertex 3
    m_linac.X_jaw_index = NULL;        // Index to acces to a jaw
    m_linac.X_jaw_nb_triangles = NULL; // Nb of triangles within each jaw
    m_linac.X_jaw_aabb = NULL;         // Bounding box of each jaw
    m_linac.X_nb_jaw = 0;              // Number of jaws

    // Jaws Y
    m_linac.Y_jaw_v1 = NULL;           // Vertex 1  - Triangular meshes
    m_linac.Y_jaw_v2 = NULL;           // Vertex 2
    m_linac.Y_jaw_v3 = NULL;           // Vertex 3
    m_linac.Y_jaw_index = NULL;        // Index to acces to a jaw
    m_linac.Y_jaw_nb_triangles = NULL; // Nb of triangles within each jaw
    m_linac.Y_jaw_aabb = NULL;         // Bounding box of each jaw
    m_linac.Y_nb_jaw = 0;              // Number of jaws

    set_name( "MeshPhanLINACNav" );
    m_mlc_filename = "";
    m_jaw_x_filename = "";
    m_jaw_y_filename = "";
    m_beam_config_filename = "";

    m_pos_mlc = make_f32xyz_zeros();
    m_loc_pos_jaw_x = make_f32xyz_zeros();
    m_loc_pos_jaw_y = make_f32xyz_zeros();
    m_rot_linac = make_f32xyz_zeros();
    m_axis_linac = make_f32matrix33_zeros();
    m_transform_linac = make_f32matrix44_zeros();

    m_beam_index = 0;
    m_field_index = 0;

}

//// Mandatory functions

void MeshPhanLINACNav::track_to_in( Particles particles )
{
/*
    if ( m_params.data_h.device_target == CPU_DEVICE )
    {
        ui32 id=0;
        while ( id<particles.size )
        {
            VPDN::kernel_host_track_to_in ( particles.data_h, m_phantom.data_h.xmin, m_phantom.data_h.xmax,
                                            m_phantom.data_h.ymin, m_phantom.data_h.ymax,
                                            m_phantom.data_h.zmin, m_phantom.data_h.zmax,
                                            m_params.data_h.geom_tolerance,
                                            id );
            ++id;
        }
    }
    else if ( m_params.data_h.device_target == GPU_DEVICE )
    {
        dim3 threads, grid;
        threads.x = m_params.data_h.gpu_block_size;
        grid.x = ( particles.size + m_params.data_h.gpu_block_size - 1 ) / m_params.data_h.gpu_block_size;

        VPIORTN::kernel_device_track_to_in<<<grid, threads>>> ( particles.data_d, m_phantom.data_d.xmin, m_phantom.data_d.xmax,
                                                                               m_phantom.data_d.ymin, m_phantom.data_d.ymax,
                                                                               m_phantom.data_d.zmin, m_phantom.data_d.zmax,
                                                                               m_params.data_d.geom_tolerance );
        cuda_error_check ( "Error ", " Kernel_VoxPhanIORT (track to in)" );
        cudaThreadSynchronize();
    }
*/
}

void MeshPhanLINACNav::track_to_out( Particles particles )
{
/*
    //
    if ( m_params.data_h.device_target == CPU_DEVICE )
    {
        VPIORTN::kernel_host_track_to_out( particles.data_h, m_phantom.data_h,
                                           m_materials.data_h, m_cross_sections.photon_CS.data_h,
                                           m_params.data_h, m_dose_calculator.dose,
                                           m_mu_table, m_hist_map );

        // Apply seTLE: splitting and determinstic raycasting
        if( m_flag_TLE == seTLE )
        {
            f64 t_start = get_time();
            m_compress_history_map();
            GGcout_time ( "Compress history map", get_time()-t_start );

            t_start = get_time();
            VPIORTN::kernel_host_seTLE( particles.data_h, m_phantom.data_h,
                                        m_coo_hist_map, m_dose_calculator.dose,
                                        m_mu_table, 100, 0.0 *eV );
            GGcout_time ( "Raycast", get_time()-t_start );
            GGnewline();

        }

    }
    else if ( m_params.data_h.device_target == GPU_DEVICE )
    {       
        dim3 threads, grid;
        threads.x = m_params.data_h.gpu_block_size;
        grid.x = ( particles.size + m_params.data_h.gpu_block_size - 1 ) / m_params.data_h.gpu_block_size;
        VPIORTN::kernel_device_track_to_out<<<grid, threads>>> ( particles.data_d, m_phantom.data_d, m_materials.data_d,
                                                              m_cross_sections.photon_CS.data_d,
                                                              m_params.data_d, m_dose_calculator.dose,
                                                              m_mu_table, m_hist_map );
        cuda_error_check ( "Error ", " Kernel_VoxPhanDosi (track to out)" );             
        cudaThreadSynchronize();

        // Apply seTLE: splitting and determinstic raycasting
        if( m_flag_TLE == seTLE )
        {
            f64 t_start = get_time();
            m_compress_history_map();
            GGcout_time ( "Compress history map", get_time()-t_start );

            threads.x = m_params.data_h.gpu_block_size;//
            grid.x = ( m_coo_hist_map.nb_data + m_params.data_h.gpu_block_size - 1 ) / m_params.data_h.gpu_block_size;

            t_start = get_time();
            VPIORTN::kernel_device_seTLE<<<grid, threads>>> ( particles.data_d, m_phantom.data_d,
                                                              m_coo_hist_map, m_dose_calculator.dose,
                                                              m_mu_table, 1000, 0.0 *eV );
            cuda_error_check ( "Error ", " Kernel_device_seTLE" );

            cudaThreadSynchronize();
            GGcout_time ( "Raycast", get_time()-t_start );
            GGnewline();
        }
    }
*/
}

void MeshPhanLINACNav::initialize( GlobalSimulationParameters params )
{
    // Check params
    if ( m_mlc_filename == "" )
    {
        GGcerr << "No mesh file specified for MLC of the LINAC phantom!" << GGendl;
        exit_simulation();
    }

    if ( m_linac.A_nb_leaves == 0 && m_linac.B_nb_leaves == 0 )
    {
        GGcerr << "MeshPhanLINACNav: number of leaves per bank must be specified!" << GGendl;
        exit_simulation();
    }

    // Params
    m_params = params;

    // Init MLC
    m_init_mlc();


    // If jaw x is defined, init
    if ( m_jaw_x_filename != "" )
    {
        m_init_jaw_x();

        // move the jaw relatively to the mlc (local frame)
        m_translate_jaw_x( 0, m_loc_pos_jaw_x );
        m_translate_jaw_x( 1, m_loc_pos_jaw_x );

    }

    // If jaw y is defined, init
    if ( m_jaw_y_filename != "" )
    {
        m_init_jaw_y();

        // move the jaw relatively to the mlc (local frame)
        m_translate_jaw_x( 0, m_loc_pos_jaw_y );
        m_translate_jaw_x( 1, m_loc_pos_jaw_y );
    }

    // Configure the linac
    m_configure_linac();


    /*
    // Check params
    if ( !m_check_mandatory() )
    {
        print_error ( "VoxPhanIORT: missing parameters." );
        exit_simulation();
    }

    // Params
    m_params = params;

    // Phantom
    m_phantom.set_name( "VoxPhanIORTNav" );
    m_phantom.initialize( params );

    // Materials table
    m_materials.load_materials_database( m_materials_filename );
    m_materials.initialize( m_phantom.list_of_materials, params );    

    // Cross Sections
    m_cross_sections.initialize( m_materials, params );

    // Init dose map
    m_dose_calculator.set_voxelized_phantom( m_phantom );
    m_dose_calculator.set_materials( m_materials );
    m_dose_calculator.set_dosel_size( m_dosel_size_x, m_dosel_size_y, m_dosel_size_z );
    m_dose_calculator.set_voi( m_xmin, m_xmax, m_ymin, m_ymax, m_zmin, m_zmax );
    m_dose_calculator.initialize( m_params ); // CPU&GPU

    // If TLE init mu and mu_en table
    if ( m_flag_TLE )
    {
        m_init_mu_table();
    }

    // if seTLE init history map
    if ( m_flag_TLE == seTLE )
    {
        HANDLE_ERROR( cudaMallocManaged( &(m_hist_map.interaction), m_phantom.data_h.number_of_voxels * sizeof( ui32 ) ) );
        HANDLE_ERROR( cudaMallocManaged( &(m_hist_map.energy), m_phantom.data_h.number_of_voxels * sizeof( f32 ) ) );

        ui32 i=0; while (i < m_phantom.data_h.number_of_voxels )
        {
            m_hist_map.interaction[ i ] = 0;
            m_hist_map.energy[ i ] = 0.0;
            ++i;
        }
    }

    // Some verbose if required
    if ( params.data_h.display_memory_usage )
    {
        ui64 mem = m_get_memory_usage();
        GGcout_mem("VoxPhanIORTNav", mem);
    }
    */
}




#endif
