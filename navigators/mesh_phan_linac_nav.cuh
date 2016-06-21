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
#include "transport_navigator.cuh"

#define HIT_JAW_X1 0
#define HIT_JAW_X2 1
#define HIT_JAW_Y1 2
#define HIT_JAW_Y2 3
#define HIT_BANK_A 4
#define HIT_BANK_B 5
#define HIT_NOTHING 6

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

    // Jaws X
    f32xyz   *X_jaw_v1;           // Vertex 1  - Triangular meshes
    f32xyz   *X_jaw_v2;           // Vertex 2
    f32xyz   *X_jaw_v3;           // Vertex 3
    ui32     *X_jaw_index;        // Index to acces to a jaw
    ui32     *X_jaw_nb_triangles; // Nb of triangles within each jaw
    AabbData *X_jaw_aabb;         // Bounding box of each jaw
    ui32      X_nb_jaw;           // Number of jaws

    // Jaws Y
    f32xyz   *Y_jaw_v1;           // Vertex 1  - Triangular meshes
    f32xyz   *Y_jaw_v2;           // Vertex 2
    f32xyz   *Y_jaw_v3;           // Vertex 3
    ui32     *Y_jaw_index;        // Index to acces to a jaw
    ui32     *Y_jaw_nb_triangles; // Nb of triangles within each jaw
    AabbData *Y_jaw_aabb;         // Bounding box of each jaw
    ui32      Y_nb_jaw;           // Number of jaws

    // Global AABB
    AabbData aabb;                // Global bounding box

    // Transformation matrix
    f32matrix44 transform;
};


// MeshPhanLINACNav -> MPLINACN
namespace MPLINACN
{
    // Device Kernel that move particles to the linac volume boundary
    __global__ void kernel_device_track_to_in( ParticlesData particles, LinacData linac, f32 geom_tolerance );

    // Host Kernel that move particles to the linac volume boundary
    void kernel_host_track_to_in( ParticlesData particles, LinacData linac, f32 geom_tolerance, ui32 id );


    __global__ void kernel_device_track_to_out( ParticlesData particles, LinacData linac,
                                                GlobalSimulationParametersData parameters );

    void kernel_host_track_to_out( ParticlesData particles, LinacData linac,
                                   GlobalSimulationParametersData parameters, ui32 id );

    __host__ __device__ void track_to_out ( ParticlesData &particles, LinacData linac,
                                            GlobalSimulationParametersData parameters,
                                            ui32 part_id );
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
    void set_jaw_x_meshes( std::string filename );
    void set_jaw_y_meshes( std::string filename );
    void set_beam_configuration( std::string filename, ui32 beam_index, ui32 field_index );

    void set_number_of_leaves( ui32 nb_bank_A, ui32 nb_bank_B );

    void set_mlc_position( f32 px, f32 py, f32 pz );

//    void set_linac_rotation( f32 rx, f32 ry, f32 rz );

    void set_linac_local_axis( f32 m00, f32 m01, f32 m02,
                               f32 m10, f32 m11, f32 m12,
                               f32 m20, f32 m21, f32 m22 );

    void set_local_jaw_x_position( f32 px, f32 py, f32 pz );

    //void set_jaw_x_rotation( f32 rx, f32 ry, f32 rz );

//    void set_jaw_x_local_axis( f32 m00, f32 m01, f32 m02,
//                             f32 m10, f32 m11, f32 m12,
//                             f32 m20, f32 m21, f32 m22 );

    void set_local_jaw_y_position( f32 px, f32 py, f32 pz );

//    void set_jaw_y_rotation( f32 rx, f32 ry, f32 rz );

//    void set_jaw_y_local_axis( f32 m00, f32 m01, f32 m02,
//                               f32 m10, f32 m11, f32 m12,
//                               f32 m20, f32 m21, f32 m22 );

    LinacData get_linac_geometry();
    f32matrix44 get_linac_transformation();

//    void set_materials( std::string filename );

private:
    void m_init_mlc();
    void m_init_jaw_x();
    void m_init_jaw_y();

    void m_translate_jaw_x( ui32 index, f32xyz T );
    void m_translate_jaw_y( ui32 index, f32xyz T );
    void m_translate_leaf_A( ui32 index, f32xyz T );
    void m_translate_leaf_B( ui32 index, f32xyz T );

    void m_configure_linac();

    std::vector< std::string > m_split_txt(std::string line);

private:

    LinacData m_linac;
    //Materials m_materials;
    //CrossSections m_cross_sections;

    // Get the memory usage
    ui64 m_get_memory_usage();

    GlobalSimulationParameters m_params;
    std::string m_materials_filename;
    std::string m_beam_config_filename;
    std::string m_mlc_filename;
    std::string m_jaw_x_filename;
    std::string m_jaw_y_filename;

    f32xyz m_pos_mlc;
    f32xyz m_loc_pos_jaw_x;
    f32xyz m_loc_pos_jaw_y;
    f32xyz m_rot_linac;
    f32matrix33 m_axis_linac;

    ui32 m_beam_index;
    ui32 m_field_index;
//
};

#endif
