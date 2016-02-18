// GGEMS Copyright (C) 2015

/*!
 * \file range_cut.cuh
 * \brief Range to energy cut converter
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 4 february 2016
 *
 * From G4RToEConvForGamma, G4RToEConvForElectron, G4VRangeToEnergyConverter,
 *      G4PhysicsTable, G4PhysicsLogVector, G4PhysicsVector
 *
 */

#ifndef RANGE_CUT_CUH
#define RANGE_CUT_CUH

#include "global.cuh"

#define NB_ELEMENTS 47

// Helper that handle log table
class LogEnergyTable
{
    public:
        LogEnergyTable( f32 theEmin, f32 theEmax, size_t theNbin );
        ~LogEnergyTable() {}

        void put_value( size_t index, f32 value);
        f32 get_energy( size_t index );
        f32 get_value( size_t index );
        f32 get_low_edge_energy( size_t index );
        size_t find_bin_location( f32 theEnergy );
        size_t find_bin( f32 e, size_t idx );
        f32 interpolation( size_t idx, f32 e );
        f32 value( f32 energy );

    private:
        f32 edgeMin;           // Energy of first point
        f32 edgeMax;           // Energy of the last point
        size_t numberOfNodes;
        std::vector<f32>  dataVector;    // Vector to keep the crossection/energyloss
        std::vector<f32>  binVector;     // Vector to keep energy
        f32 dBin;          // Bin width - useful only for fixed binning
        f32 baseBin;       // Set this in constructor for performance
};


class RangeCut
{
    public:
        RangeCut();
        ~RangeCut() {}

        // calculate energy cut from given range cut for the material
        //f32 convert_gamma(f32 rangeCut, const MaterialsTable* material, ui32 mat_id);
        f32 convert_gamma(f32 rangeCut, const ui16* mixture, ui16 nb_elts,
                          const f32 *atom_num_dens, ui32 abs_index);

        f32 convert_electron(f32 rangeCut, const ui16 *mixture, ui16 nb_elts,
                             const f32 *atom_num_dens, f32 density, ui32 abs_index);

    private:
        // Global
        ui32 TotBin;
        f32  LowestEnergy;
        f32  HighestEnergy;
        f32  MaxEnergyCut;

        // For gamma
        LogEnergyTable* gamma_range_table;
        f32 compute_gamma_cross_sections( f32 AtomicNumber, f32 KineticEnergy );
        void build_gamma_cross_sections_table();
        void gamma_build_range_table( const ui16* mixture, ui16 nb_elts, const f32 *atom_num_dens, ui32 abs_index,
                                      LogEnergyTable *range_table );

        std::vector< LogEnergyTable* > gamma_CS_table;
        f32 gZ;
        f32 s200keV, s1keV;
        f32 tmin, tlow;
        f32 smin, slow;
        f32 cmin, clow, chigh;

        // For e-
        LogEnergyTable* electron_range_table;
        f32 compute_electron_loss( f32 AtomicNumber, f32 KineticEnergy );
        void build_electron_loss_table();
        void electron_build_range_table(const ui16* mixture, ui16 nb_elts, const f32 *atom_num_dens, ui32 abs_index,
                                        LogEnergyTable* range_table );

        std::vector< LogEnergyTable* > e_loss_table;
        f32 Mass;
        f32 eZ;
        f32 taul;
        f32 ionpot;
        f32 ionpotlog;
        f32 bremfactor;

        // Converter
        f32 convert_cut_to_energy(LogEnergyTable* rangeVector, f32 theCutInLength);
};

#endif
