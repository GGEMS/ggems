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

#ifndef MESH_PHAN_LINAC_NAV_CUH
#define MESH_PHAN_LINAC_NAV_CUH

#include "global.cuh"
#include "ggems_phantom.cuh"
#include "particles.cuh"
#include "primitives.cuh"
#include "mesh_io.cuh"

// LINAC data (GPU' proof but not a complete SoA support)
struct LinacData
{
    // Leaves in Bank A
    f32xyz   *A_leaf_v1;           // Vertex 1  - Triangular meshes
    f32xyz   *A_leaf_v2;           // Vertex 2
    f32xyz   *A_leaf_v3;           // Vertex 3
    ui32     *A_leaf_index;        // Index to acces to a leaf
    ui32     *A_leaf_nb_triangles; // Nb of triangles within each leaf
    AabbData *A_leaf_aabb;         // Bounding box of each leaf
    AabbData  A_bank_aabb;         // Bounding box of the bank A
    ui32      A_nb_leaves;         // Number of leaves in the bank A

    // Leaves in Bank B
    f32xyz   *B_leaf_v1;           // Vertex 1  - Triangular meshes
    f32xyz   *B_leaf_v2;           // Vertex 2
    f32xyz   *B_leaf_v3;           // Vertex 3
    ui32     *B_leaf_index;        // Index to acces to a leaf
    ui32     *B_leaf_nb_triangles; // Nb of triangles within each leaf
    AabbData *B_leaf_aabb;         // Bounding box of each leaf
    AabbData  B_bank_aabb;         // Bounding box of the bank B
    ui32      B_nb_leaves;         // Number of leaves in the bank B

    // Backup X
    // TODO

    // Backup Y
    // TODO

};


// MeshPhanLINACNav -> MPLINACN
namespace MPLINACN
{


}

class MeshPhanLINACNav : public GGEMSPhantom
{
public:
    MeshPhanLINACNav();
    ~MeshPhanLINACNav() {}

    // Init
    void initialize( GlobalSimulationParameters params );
    // Tracking from outside to the phantom border
    void track_to_in( Particles particles );
    // Tracking inside the phantom until the phantom border
    void track_to_out( Particles particles );

    void set_mlc_meshes( std::string filename );
    void set_number_of_leaves( ui32 nb_bank_A, ui32 nb_bank_B );

    LinacData get_linac_geometry();

//    void set_materials( std::string filename );

private:
    void m_init_mlc();

private:

    LinacData m_linac;
    //Materials m_materials;
    //CrossSections m_cross_sections;


    // Get the memory usage
    ui64 m_get_memory_usage();


    GlobalSimulationParameters m_params;
    std::string m_materials_filename;
    std::string m_mlc_filename;

//
};

#endif
