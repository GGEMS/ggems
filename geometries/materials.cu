// GGEMS Copyright (C) 2015

#ifndef MATERIALS_CU
#define MATERIALS_CU

#include "materials.cuh"


//////////////////////////////////////////////////////////////////
//// Density Effect Data /////////////////////////////////////////
//////////////////////////////////////////////////////////////////

static const ui8 element_state[ 96 ] =
{
    0,                                                               // 0
    STATE_GAS,   STATE_GAS,   STATE_SOLID, STATE_SOLID, STATE_SOLID, // 1-5
    STATE_SOLID, STATE_GAS,   STATE_GAS,   STATE_GAS,   STATE_GAS,   // 6-10
    STATE_SOLID, STATE_SOLID, STATE_SOLID, STATE_SOLID, STATE_SOLID, // 11-15
    STATE_SOLID, STATE_GAS,   STATE_GAS,   STATE_SOLID, STATE_SOLID, // 16-20
    STATE_SOLID, STATE_SOLID, STATE_SOLID, STATE_SOLID, STATE_SOLID, // 21-25
    STATE_SOLID, STATE_SOLID, STATE_SOLID, STATE_SOLID, STATE_SOLID, // 26-30
    STATE_SOLID, STATE_SOLID, STATE_SOLID, STATE_SOLID, STATE_GAS,   // 31-35
    STATE_GAS,   STATE_SOLID, STATE_SOLID, STATE_SOLID, STATE_SOLID, // 36-40
    STATE_SOLID, STATE_SOLID, STATE_SOLID, STATE_SOLID, STATE_SOLID, // 41-45
    STATE_SOLID, STATE_SOLID, STATE_SOLID, STATE_SOLID, STATE_SOLID, // 46-50
    STATE_SOLID, STATE_SOLID, STATE_SOLID, STATE_GAS,   STATE_SOLID, // 51-55
    STATE_SOLID, STATE_SOLID, STATE_SOLID, STATE_SOLID, STATE_SOLID, // 56-60
    STATE_SOLID, STATE_SOLID, STATE_SOLID, STATE_SOLID, STATE_SOLID, // 61-65
    STATE_SOLID, STATE_SOLID, STATE_SOLID, STATE_SOLID, STATE_SOLID, // 66-70
    STATE_SOLID, STATE_SOLID, STATE_SOLID, STATE_SOLID, STATE_SOLID, // 71-75
    STATE_SOLID, STATE_SOLID, STATE_SOLID, STATE_SOLID, STATE_SOLID, // 76-80
    STATE_SOLID, STATE_SOLID, STATE_SOLID, STATE_SOLID, STATE_SOLID, // 81-85
    STATE_GAS,   STATE_SOLID, STATE_SOLID, STATE_SOLID, STATE_SOLID, // 86-90
    STATE_SOLID, STATE_SOLID, STATE_SOLID, STATE_SOLID, STATE_SOLID, // 91-95
};


static const f32 density_effect_data[ 96 ][ 9] =
{
    // Eplasma, rho, -C, X_0, X_1, m, a, delta_0, delta_max
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},  // Z=0
    {0.263, 1.412, 9.5835, 1.8639, 3.2718, 0.14092, 5.7273, 0.0, 0.024},
    {0.263, 1.7, 11.1393, 2.2017, 3.6122, 0.13443, 5.8347, 0, 0.024},
    {13.844, 1.535, 3.1221, 0.1304, 1.6397, 0.95136, 2.4993, 0.14, 0.062},
    {26.096, 1.908, 2.7847, 0.0392, 1.6922, 0.80392, 2.4339, 0.14, 0.029},
    {30.17, 2.32, 2.8477, 0.0305, 1.9688, 0.56224, 2.4512, 0.14, 0.024},
    {28.803, 2.376, 2.9925, -0.0351, 2.486, 0.2024, 3.0036, 0.1, 0.038},
    {0.695, 1.984, 10.5400, 1.7378, 4.1323, 0.15349, 3.2125, 0.0, 0.086},
    {0.744, 2.314, 10.7004, 1.7541, 4.3213, 0.11778, 3.2913, 0.0, 0.101},
    {0.788, 2.450, 10.9653, 1.8433, 4.4096, 0.11083, 3.2962, 0.0, 0.121},
    {0.587, 2.577, 11.9041, 2.0735, 4.6421, 0.08064, 3.5771, 0.0, 0.110},
    {19.641, 2.648, 5.0526, 0.2880, 3.1962, 0.07772, 3.6452, 0.08, 0.098},
    {26.708, 2.331, 4.5297, 0.1499, 3.0668, 0.081163, 3.6166, 0.08, 0.073},
    {32.86, 2.18, 4.2395, 0.1708, 3.0127, 0.08024, 3.6345, 0.12, 0.061},
    {31.055, 2.103, 4.4351, 0.2014, 2.8715, 0.14921, 3.2546, 0.14, 0.059},
    {29.743, 2.056, 4.5214, 0.1696, 2.7815, 0.2361, 2.9158, 0.14, 0.057},
    {28.789, 2.131, 4.6659, 0.158, 2.7159, 0.33992, 2.6456, 0.14, 0.059},
    {1.092, 1.734, 11.1421, 1.5555, 4.2994, 0.19849, 2.9702, 0.0, 0.041},
    {0.789, 1.753, 11.9480, 1.7635, 4.4855, 0.19714, 2.9618, 0.0, 0.037},
    {18.65, 1.830, 5.6423, 0.3851, 3.1724, 0.19827, 2.9233, 0.10, 0.035},
    {25.342, 1.666, 5.0396, 0.3228, 3.1191, 0.15643, 3.0745, 0.14, 0.031},
    {34.050, 1.826, 4.6949, 0.1640, 3.0593, 0.15754, 3.0517, 0.10, 0.027},
    {41.619, 1.969, 4.4450, 0.0957, 3.0386, 0.15662, 3.0302, 0.12, 0.025},
    {47.861, 2.070, 4.2659, 0.0691, 3.0322, 0.15436, 3.0163, 0.14, 0.024},
    {52.458, 2.181, 4.1781, 0.0340, 3.0451, 0.15419, 2.9896, 0.14, 0.023},
    {53.022, 2.347, 4.2702, 0.0447, 3.1074, 0.14973, 2.9796, 0.14, 0.021},
    {55.172, 2.504, 4.2911, -0.0012, 3.1531, 0.146, 2.9632, 0.12, 0.021},
    {58.188, 2.626, 4.2601, -0.0187, 3.1790, 0.14474, 2.9502, 0.12, 0.019},
    {59.385, 2.889, 4.3115, -0.0566, 3.1851, 0.16496, 2.843, 0.10, 0.020},
    {58.270, 2.956, 4.4190, -0.0254, 3.2792, 0.14339, 2.9044, 0.08, 0.019},
    {52.132, 3.142, 4.6906, 0.0049, 3.3668, 0.14714, 2.8652, 0.08, 0.019},
    {46.688, 2.747, 4.9353, 0.2267, 3.5434, 0.09440, 3.1314, 0.14, 0.019},
    {44.141, 2.461, 5.1411, 0.3376, 3.6096, 0.07188, 3.3306, 0.14, 0.025},
    {45.779, 2.219, 5.0510, 0.1767, 3.5702, 0.06633, 3.4176, 0.00, 0.030},
    {40.112, 2.104, 5.3210, 0.2258, 3.6264, 0.06568, 3.4317, 0.10, 0.024},
    {1.604, 1.845, 11.7307, 1.5262, 4.9899, 0.06335, 3.467, 0, 0.022},
    {1.114, 1.77, 12.5115, 1.7158, 5.0748, 0.07446, 3.4051, 0, 0.025},
    {23.467, 1.823, 6.4776, 0.5737, 3.7995, 0.07261, 3.4177, 0.14, 0.026},
    {30.244, 1.707, 5.9867, 0.4585, 3.6778, 0.07165, 3.4435, 0.14, 0.026},
    {40.346, 1.649, 5.4801, 0.3608, 3.5542, 0.07138, 3.4565, 0.14, 0.027},
    {48.671, 1.638, 5.1774, 0.2957, 3.489, 0.07177, 3.4533, 0.14, 0.028},
    {56.039, 1.734, 5.0141, 0.1785, 3.2201, 0.13883, 3.093, 0.14, 0.036},
    {60.951, 1.658, 4.8793, 0.2267, 3.2784, 0.10525, 3.2549, 0.14, 0.03},
    {64.760, 1.727, 4.7769, 0.0949, 3.1253, 0.16572, 2.9738, 0.14, 0.040},
    {66.978, 1.780, 4.7694, 0.0599, 3.0834, 0.19342, 2.8707, 0.14, 0.046},
    {67.128, 1.804, 4.8008, 0.0576, 3.1069, 0.19205, 2.8633, 0.14, 0.046},
    {65.683, 1.911, 4.9358, 0.0563, 3.0555, 0.24178, 2.7239, 0.14, 0.047},
    {61.635, 1.933, 5.0630, 0.0657, 3.1074, 0.24585, 2.6899, 0.14, 0.052},
    {55.381, 1.895, 5.2727, 0.1281, 3.1667, 0.24609, 2.6772, 0.14, 0.051},
    {50.896, 1.851, 5.5211, 0.2406, 3.2032, 0.23879, 2.7144, 0.14, 0.044},
    {50.567, 1.732, 5.5340, 0.2879, 3.2959, 0.18689, 2.8576, 0.14, 0.037},
    {48.242, 1.645, 5.6241, 0.3189, 3.3489, 0.16652, 2.9319, 0.14, 0.034},
    {45.952, 1.577, 5.7131, 0.3296, 3.4418, 0.13815, 3.0354, 0.14, 0.033},
    {41.348, 1.498, 5.9488, 0.0549, 3.2596, 0.23766, 2.7276, 0.0, 0.045},
    {1.369, 1.435, 12.7281, 1.563, 4.7371, 0.23314, 2.7414, 0, 0.043},
    {25.37, 1.462, 6.9135, 0.5473, 3.5914, 0.18233, 2.8866, 0.14, 0.035},
    {34.425, 1.410, 6.3153, 0.4190, 3.4547, 0.18268, 2.8906, 0.14, 0.035},
    {45.792, 1.392, 5.7850, 0.3161, 3.3293, 0.18591, 2.8828, 0.14, 0.036},
    {47.834, 1.461, 5.7837, 0.2713, 3.3432, 0.18885, 2.8592, 0.14, 0.040},
    {48.301, 1.520, 5.8096, 0.2333, 3.2773, 0.23265, 2.7331, 0.14, 0.041},
    {48.819, 1.588, 5.8290, 0.1984, 3.3063, 0.23530, 2.7050, 0.14, 0.044},
    {50.236, 1.672, 5.8224, 0.1627, 3.3199, 0.24280, 2.6674, 0.14, 0.048},
    {50.540, 1.749, 5.8597, 0.1520, 3.3460, 0.24698, 2.6403, 0.14, 0.053},
    {42.484, 1.838, 6.2278, 0.1888, 3.4633, 0.24448, 2.6245, 0.14, 0.06},
    {51.672, 1.882, 5.8738, 0.1058, 3.3932, 0.25109, 2.5977, 0.14, 0.061},
    {52.865, 1.993, 5.9045, 0.0947, 3.4224, 0.24453, 2.6056, 0.14, 0.063},
    {53.698, 2.081, 5.9183, 0.0822, 3.4474, 0.24665, 2.5849, 0.14, 0.061},
    {54.467, 2.197, 5.9587, 0.0761, 3.4782, 0.24638, 2.5726, 0.14, 0.062},
    {55.322, 2.26, 5.9521, 0.0648, 3.4922, 0.24823, 2.5573, 0.14, 0.061},
    {56.225, 2.333, 5.9677, 0.0812, 3.5085, 0.24189, 2.5469, 0.14, 0.062},
    {47.546, 2.505, 6.3325, 0.1199, 3.6246, 0.25295, 2.5141, 0.14, 0.071},
    {57.581, 2.348, 5.9785, 0.1560, 3.5218, 0.24033, 2.5643, 0.14, 0.054},
    {66.770, 2.174, 5.7139, 0.1965, 3.4337, 0.22918, 2.6155, 0.14, 0.035},
    {74.692, 2.07, 5.5262, 0.2117, 3.4805, 0.17798, 2.7623, 0.14, 0.03},
    {80.315, 1.997, 5.4059, 0.2167, 3.496, 0.15509, 2.8447, 0.14, 0.027},
    {83.846, 1.976, 5.3445, 0.0559, 3.4845, 0.15184, 2.8627, 0.08, 0.026},
    {86.537, 1.947, 5.3083, 0.0891, 3.5414, 0.12751, 2.9608, 0.10, 0.023},
    {86.357, 1.927, 5.3418, 0.0819, 3.5480, 0.12690, 2.9658, 0.10, 0.023},
    {84.389, 1.965, 5.4732, 0.1484, 3.6212, 0.11128, 3.0417, 0.12, 0.021},
    {80.215, 1.926, 5.5747, 0.2021, 3.6979, 0.09756, 3.1101, 0.14, 0.020},
    {66.977, 1.904, 5.9605, 0.2756, 3.7275, 0.11014, 3.0519, 0.14, 0.021},
    {62.104, 1.814, 6.1365, 0.3491, 3.8044, 0.09455, 3.1450, 0.14, 0.019},
    {61.072, 1.755, 6.2018, 0.3776, 3.8073, 0.09359, 3.1608, 0.14, 0.019},
    {56.696, 1.684, 6.3505, 0.4152, 3.8248, 0.0941, 3.1671, 0.14, 0.02},
    {55.773, 1.637, 6.4003, 0.4267, 3.8293, 0.09282, 3.183, 0.14, 0.02},
    {1.708, 1.458, 13.2839, 1.5368, 4.9889, 0.20798, 2.7409, 0, 0.057},
    {40.205, 1.403, 7.0452, 0.5991, 3.9428, 0.08804, 3.2454, 0.14, 0.022},
    {57.254, 1.380, 6.3742, 0.4559, 3.7966, 0.08567, 3.2683, 0.14, 0.023},
    {61.438, 1.363, 6.2473, 0.4202, 3.7681, 0.08655, 3.2610, 0.14, 0.025},
    {70.901, 1.42, 6.0327, 0.3144, 3.5079, .14770, 2.9845, 0.14, 0.036},
    {77.986, 1.447, 5.8694, 0.2260, 3.3721, .19677, 2.8171, 0.14, 0.043},
    {81.221, 1.468, 5.8149, 0.1869, 3.369, 0.19741, 2.8082, 0.14, 0.043},
    {80.486, 1.519, 5.8748, 0.1557, 3.3981, 0.20419, 2.7679, 0.14, 0.057},
    {66.607, 1.552, 6.2813, 0.2274, 3.5021, 0.20308, 2.7615, 0.14, 0.056},
    {66.022, 1.559, 6.3097, 0.2484, 3.516, .20257, 2.7579, 0.14, 0.056},
    {67.557, 1.574, 6.2912, 0.2378, 3.5186, .20192, 2.7560, 0.14, 0.062}
};

//////////////////////////////////////////////////////////////////
//// MaterialDataBase class //////////////////////////////////////
//////////////////////////////////////////////////////////////////

MaterialsDataBase::MaterialsDataBase() {} // Open and load the material database


//// Private ///////////////

// Add elements into the table
void MaterialsDataBase::m_add_elements( std::string elt_name, ui16 elt_Z, f32 elt_A, f32 elt_pot )
{
    elements_Z[ elt_name ] = elt_Z;
    elements_A[ elt_name ] = elt_A *gram/mole;
    elements_pot[ elt_name ] = elt_pot *eV;   
}

//// Public ///////////////

// Load materials from data base
void MaterialsDataBase::load_materials(std::string filename) {
    //printf("load material ... \n");
    std::ifstream file(filename.c_str());

    std::string line, elt_name;
    f32 mat_f;
    ui16 i;
    ui16 ind = 0;

    while (file) {
        m_txt_reader.skip_comment(file);
        std::getline(file, line);

        if (file) {
            aMaterial mat;
            mat.name = m_txt_reader.read_material_name(line);
            //printf("mat name ... %s \n",mat.name.c_str());   // Too much verbose - JB
            mat.density = m_txt_reader.read_material_density(line);
            mat.nb_elements = m_txt_reader.read_material_nb_elements(line);

            i=0; while (i<mat.nb_elements) {
                std::getline(file, line);
                elt_name = m_txt_reader.read_material_element(line);
                mat_f = m_txt_reader.read_material_fraction(line);

                mat.mixture_Z.push_back(elt_name);
                mat.mixture_f.push_back(mat_f);

                ++i;
            }

            materials[mat.name] = mat;
            ++ind;

        } // if

    } // while

}

// Get density
f32 MaterialsDataBase::get_density( std::string mat_name )
{
    return materials[ mat_name ].density;
}

// Get nb of elements
ui16 MaterialsDataBase::get_nb_elements( std::string mat_name )
{
    return materials[ mat_name ].nb_elements;
}

// Get elements name
std::string MaterialsDataBase::get_element_name( std::string mat_name, ui16 index )
{
    return materials[ mat_name ].mixture_Z[ index ];
}

// Get atom num dens
f32 MaterialsDataBase::get_atom_num_dens( std::string mat_name, ui16 index )
{
    std::string elt_name = get_element_name( mat_name, index );
    return Avogadro / elements_A[ elt_name ] *
            materials[ mat_name ].mixture_f[ index ] * get_density( mat_name );
}

// Get Z
ui16 MaterialsDataBase::get_element_Z( std::string elt_name )
{
    return elements_Z[ elt_name ];
}

// Get A
f32 MaterialsDataBase::get_element_A( std::string elt_name )
{
    return elements_A[ elt_name ];
}

// Get mean excitation energy
f32 MaterialsDataBase::get_element_pot( std::string elt_name )
{
    return elements_pot[ elt_name ];
}

// Compute Ioni params
void MaterialsDataBase::compute_ioni_parameters( std::string mat_name )
{
    /// Compute Mean Excitation Energy ///////////////////////////
    m_LogMeanExcEnergy = 0;
    m_TotNbOfElectPerVolume = 0;
    m_MeanExcEnergy = 0;

    ui32 nelm = get_nb_elements( mat_name );

    std::string elt_name;
    f32 AxZ;

    for ( size_t i = 0; i < nelm; i++)
    {
        elt_name = get_element_name( mat_name, i );
        AxZ = get_atom_num_dens( mat_name, i ) * ( f32 )get_element_Z( elt_name );
        m_LogMeanExcEnergy += AxZ * logf( get_element_pot( elt_name ) );
        m_TotNbOfElectPerVolume += AxZ;
    }

    m_LogMeanExcEnergy /= m_TotNbOfElectPerVolume;
    m_MeanExcEnergy = expf( m_LogMeanExcEnergy );

    /// Compute density effect for correction factor //////////////

    m_X0 = 0.0; m_X1 = 0.0; m_C = 0.0; m_A = 0.0; m_M = 0.0; m_D0 = 0.0;

    // define material state (approximation based on threshold)
    ui8 state = STATE_GAS;
    if ( get_density( mat_name ) > kGasThreshold ) state = STATE_SOLID;

    // Check if density effect data exist in the table
    // R.M. Sternheimer, Atomic Data and Nuclear Data Tables, 30: 261 (1984)

    ui16 Z0 = get_element_Z( get_element_name( mat_name, 0 ) );

    if ( nelm == 1 && state == get_element_state( Z0 ) ) {

        // Take parameters for the density effect correction from
        // R.M. Sternheimer et al. Density Effect For The Ionization Loss
        // of Charged Particles in Various Substances.
        // Atom. Data Nucl. Data Tabl. 30 (1984) 261-271.

        m_C = m_read_C_density( Z0 );
        m_M = m_read_M_density( Z0 );
        m_A = m_read_A_density( Z0 );
        m_X0 = m_read_X0_density( Z0 );
        m_X1 = m_read_X1_density( Z0 );
        m_D0 = m_read_D0_density( Z0 );
        //f32 fPlasmaEnergy = fDensityData->GetPlasmaEnergy(idx);
        //f32 fAdjustmentFactor = fDensityData->GetAdjustmentFactor(idx);

        /* No base material construction in GGEMS - JB
        // Correction for base material
        const G4Material* bmat = fMaterial->GetBaseMaterial();
        if(bmat) {
            G4double corr = G4Log(bmat->GetDensity()/fMaterial->GetDensity());
            fCdensity  += corr;
            fX0density += corr/twoln10;
            fX1density += corr/twoln10;
        }
        */

    } else {

        static const f32 Cd2 = 4*pi*hbarc_squared*classic_electr_radius;
        f32 fPlasmaEnergy = sqrtf( Cd2*m_TotNbOfElectPerVolume );

        // Compute parameters for the density effect correction in DE/Dx formula.
        // The parametrization is from R.M. Sternheimer, Phys. Rev.B,3:3681 (1971)
        ui8 icase;

        m_C = 1. + 2*logf( m_MeanExcEnergy / fPlasmaEnergy);
        //
        // condensed materials
        //
        if ( state == STATE_SOLID )   // ||(State == kStateLiquid)) {
        {
            static const f32 E100eV  = 100.*eV;
            static const f32 ClimiS[] = { 3.681 , 5.215 };
            static const f32 X0valS[] = { 1.0   , 1.5   };
            static const f32 X1valS[] = { 2.0   , 3.0   };

            if (m_MeanExcEnergy < E100eV)
            {
                icase = 0;
            }
            else
            {
                icase = 1;
            }

            if ( m_C < ClimiS[ icase ] )
            {
                m_X0 = 0.2;
            }
            else
            {
                m_X0 = 0.326 * m_C - X0valS[ icase ];
            }

            m_X1 = X1valS[ icase ]; m_M = 3.0;

            //special: Hydrogen (liquid)
            if ( nelm == 1 && Z0 == 1 )
            {
                m_X0 = 0.425; m_X1 = 2.0; m_M = 5.949;
            }
        }
        //
        // gases
        //
        if ( state == STATE_GAS )
        {
            m_M = 3.;
            m_X1 = 4.0;
            //static const G4double ClimiG[] = {10.,10.5,11.,11.5,12.25,13.804};
            //static const G4double X0valG[] = {1.6,1.7,1.8,1.9,2.0,2.0};
            //static const G4double X1valG[] = {4.0,4.0,4.0,4.0,4.0,5.0};

            if ( m_C < 10.)
            {
                m_X0 = 1.6;
            }
            else if ( m_C < 11.5)
            {
                m_X0 = 1.6 + 0.2*( m_C - 10. );
            }
            else if ( m_C < 12.25)
            {
                m_X0 = 1.9 + ( m_C - 11.5 ) / 7.5;
            } else if ( m_C < 13.804)
            {
                m_X0 = 2.0;
                m_X1 = 4.0 + ( m_C - 12.25 ) / 1.554;
            } else
            {
                m_X0 = 0.326 * m_C - 2.5; m_X1 = 5.0;
            }

            /* Should never get here ?   - JB
            //special: Hydrogen
            if ( nelm == 1 && Z0 == 1 ) {
                fX0density = 1.837; fX1density = 3.0; fMdensity = 4.754;
            }

            //special: Helium
            if (1 == nelm && 2 == Z0) {
                fX0density = 2.191; fX1density = 3.0; fMdensity = 3.297;
            }
            */
        }
    }

    /*  Not applied this correction    - JB
    // change parameters if the gas is not in STP.
    // For the correction the density(STP) is needed.
    // Density(STP) is calculated here :

    if (State == kStateGas) {
        G4double Density  = fMaterial->GetDensity();
        G4double Pressure = fMaterial->GetPressure();
        G4double Temp     = fMaterial->GetTemperature();

        G4double DensitySTP = Density*STP_Pressure*Temp/(Pressure*NTP_Temperature);

        G4double ParCorr = G4Log(Density/DensitySTP);

        fCdensity  -= ParCorr;
        fX0density -= ParCorr/twoln10;
        fX1density -= ParCorr/twoln10;
    }
    */

    // fAdensity parameter can be fixed for not conductive materials
    if ( m_D0 == 0.0 )
    {
        f32 twoln10 = 2. * logf( 10 );
        f32 Xa = m_C / twoln10;
        m_A = twoln10*( Xa - m_X0 ) / powf( ( m_X1 - m_X0 ), m_M );
    }


    /// Compute parameters for the energy loss fluctuation model //////////

    m_F1fluct = 0.0; m_F2fluct = 0.0; m_Energy0fluct = 0.0;
    m_Energy1fluct = 0.0; m_Energy2fluct = 0.0; m_LogEnergy1fluct = 0.0;
    m_LogEnergy2fluct = 0.0;

    // needs an 'effective Z'
    f32 Zeff = 0.;

    for ( size_t i = 0; i < nelm; i++ )
    {
        elt_name = get_element_name( mat_name, i );
        Zeff += ( materials[ mat_name ].mixture_f[ i ] * ( f32 )get_element_Z( elt_name ) );
    }

    if ( Zeff > 2. )
    {
        m_F2fluct = 2. / Zeff;
    }
    else
    {
        m_F2fluct = 0.;
    }

    m_F1fluct         = 1. - m_F2fluct;
    m_Energy2fluct    = 10. * Zeff*Zeff*eV;
    m_LogEnergy2fluct = logf( m_Energy2fluct );
    m_LogEnergy1fluct = ( m_LogMeanExcEnergy - m_F2fluct * m_LogEnergy2fluct) / m_F1fluct;
    m_Energy1fluct    = expf( m_LogEnergy1fluct );
    m_Energy0fluct    = 10.*eV;

    //fRateionexcfluct = 0.4;    Not used - JB
}


// Get mean electron excitation
f32 MaterialsDataBase::get_mean_excitation()
{
   return m_MeanExcEnergy;
}
f32 MaterialsDataBase::get_X0_density()
{
    return m_X0;
}
f32 MaterialsDataBase::get_X1_density()
{
    return m_X1;
}
f32 MaterialsDataBase::get_D0_density()
{
    return m_D0;
}
f32 MaterialsDataBase::get_C_density()
{
    return m_C;
}
f32 MaterialsDataBase::get_A_density()
{
    return m_A;
}
f32 MaterialsDataBase::get_M_density()
{
    return m_M;
}


f32 MaterialsDataBase::get_F1_fluct()
{
    return m_F1fluct;
}
f32 MaterialsDataBase::get_F2_fluct()
{
    return m_F2fluct;
}
f32 MaterialsDataBase::get_Energy0_fluct()
{
    return m_Energy0fluct;
}
f32 MaterialsDataBase::get_Energy1_fluct()
{
    return m_Energy1fluct;
}
f32 MaterialsDataBase::get_Energy2_fluct()
{
    return m_Energy2fluct;
}
f32 MaterialsDataBase::get_LogEnergy1_fluct()
{
    return m_LogEnergy1fluct;
}
f32 MaterialsDataBase::get_LogEnergy2_fluct()
{
    return m_LogEnergy2fluct;
}

// Get radiation length
f32 MaterialsDataBase::get_rad_len( std::string mat_name )
{
    f32 radinv = 0.0 ;
    f32 radTsai, Zeff, Coulomb;
    std::string elt_name;

    static const f32 Lrad_light[]  = {5.31  , 4.79  , 4.74 ,  4.71} ;
    static const f32 Lprad_light[] = {6.144 , 5.621 , 5.805 , 5.924} ;
    static const f32 k1 = 0.0083 , k2 = 0.20206 ,k3 = 0.0020 , k4 = 0.0369 ;

    for ( size_t i = 0; i < get_nb_elements( mat_name ); i++)
    {
        elt_name = get_element_name( mat_name, i );
        Zeff = ( f32 )get_element_Z( elt_name );

        //  Compute Coulomb correction factor (Phys Rev. D50 3-1 (1994) page 1254)
        f32 az2 = (fine_structure_const*Zeff)*(fine_structure_const*Zeff);
        f32 az4 = az2 * az2;
        Coulomb = ( k1*az4 + k2 + 1./ ( 1. + az2 ) ) * az2 - ( k3*az4 + k4 ) * az4;

        //  Compute Tsai's Expression for the Radiation Length
        //  (Phys Rev. D50 3-1 (1994) page 1254)
        const f32 logZ3 = logf( Zeff ) / 3.;

        f32 Lrad, Lprad;
        i32 iz = ( i32 ) ( Zeff + 0.5 ) - 1 ;
        if ( iz <= 3 )
        {
            Lrad = Lrad_light[ iz ];  Lprad = Lprad_light[ iz ];
        }
        else
        {
            Lrad = logf( 184.15 ) - logZ3; Lprad = logf( 1194. ) - 2*logZ3;
        }

        radTsai = 4*alpha_rcl2*Zeff * ( Zeff * ( Lrad - Coulomb ) + Lprad );
        radinv += get_atom_num_dens( mat_name, i ) * radTsai;
    }

    return ( radinv <= 0.0 ? FLT_MAX : 1. / radinv);

}

// Get eIonisation correction
f32 MaterialsDataBase::m_read_X0_density( ui16 Z )
{
    return density_effect_data[ Z ][ 3 ];
}
// Get eIonisation correction
f32 MaterialsDataBase::m_read_X1_density( ui16 Z )
{
    return density_effect_data[ Z ][ 4 ];
}
// Get eIonisation correction
f32 MaterialsDataBase::m_read_D0_density( ui16 Z )
{
    return density_effect_data[ Z ][ 7 ];
}
// Get eIonisation correction
f32 MaterialsDataBase::m_read_C_density( ui16 Z )
{
    return density_effect_data[ Z ][ 2 ];
}
// Get eIonisation correction
f32 MaterialsDataBase::m_read_A_density( ui16 Z )
{
    return density_effect_data[ Z ][ 5 ];
}
// Get eIonisation correction
f32 MaterialsDataBase::m_read_M_density( ui16 Z )
{
    return density_effect_data[ Z ][ 6 ];
}
// Get element state
ui8 MaterialsDataBase::get_element_state( ui16 Z )
{
    return element_state[ Z ];
}

// Load elements from internal data
void MaterialsDataBase::load_elements()
{
    // Clear elements data
    elements_Z.clear();
    elements_A.clear();
    elements_pot.clear();

    //            Elt name,      Z,  A (g/mole), mean ioni potential (eV)
    m_add_elements( "Hydrogen"   ,  1,   1.01  ,  19.2 );
    m_add_elements( "Helium"     ,  2,   4.003 ,  41.8 );
    m_add_elements( "Lithium"    ,  3,   6.941 ,  40.0 );
    m_add_elements( "Beryllium"  ,  4,   9.012 ,  63.7 );
    m_add_elements( "Boron"      ,  5,  10.811 ,  76.0 );
    m_add_elements( "Carbon"     ,  6,  12.01  ,  81.0 );
    m_add_elements( "Nitrogen"   ,  7,  14.01  ,  82.0 );
    m_add_elements( "Oxygen"     ,  8,  16.00  ,  95.0 );
    m_add_elements( "Fluorine"   ,  9,  18.998 , 115.0 );
    m_add_elements( "Neon"       , 10,  20.180 , 137.0 );
    m_add_elements( "Sodium"     , 11,  22.99  , 149.0 );
    m_add_elements( "Magnesium"  , 12,  24.305 , 156.0 );
    m_add_elements( "Aluminium"  , 13,  26.98  , 166.0 );
    m_add_elements( "Silicon"    , 14,  28.09  , 173.0 );
    m_add_elements( "Phosphor"   , 15,  30.97  , 173.0 );
    m_add_elements( "Sulfur"     , 16,  32.066 , 180.0 );
    m_add_elements( "Chlorine"   , 17,  35.45  , 174.0 );
    m_add_elements( "Argon"	     , 18,  39.95  , 188.0 );
    m_add_elements( "Potassium"  , 19,  39.098 , 190.0 );
    m_add_elements( "Calcium"    , 20,  40.08  , 191.0 );
    m_add_elements( "Scandium"   , 21,  44.956 , 216.0 );
    m_add_elements( "Titanium"   , 22,  47.867 , 233.0 );
    m_add_elements( "Vandium"    , 23,  50.942 , 245.0 );
    m_add_elements( "Chromium"   , 24,  51.996 , 257.0 );
    m_add_elements( "Manganese"  , 25,  54.938 , 272.0 );
    m_add_elements( "Iron"       , 26,  55.845 , 286.0 );
    m_add_elements( "Cobalt"     , 27,  58.933 , 297.0 );
    m_add_elements( "Nickel"     , 28,  58.693 , 311.0 );
    m_add_elements( "Copper"     , 29,  63.39  , 322.0 );
    m_add_elements( "Zinc"       , 30,  65.39  , 330.0 );
    m_add_elements( "Gallium"    , 31,  69.723 , 334.0 );
    m_add_elements( "Germanium"  , 32,  72.61  , 350.0 );
    m_add_elements( "Yttrium"    , 39,  88.91  , 379.0 );
    m_add_elements( "Silver"     , 47, 107.868 , 470.0 );
    m_add_elements( "Cadmium"    , 48, 112.41  , 469.0 );
    m_add_elements( "Tin"        , 50, 118.71  , 488.0 );
    m_add_elements( "Tellurium"  , 52, 127.6   , 485.0 );
    m_add_elements( "Iodine"     , 53, 126.90  , 491.0 );
    m_add_elements( "Cesium"     , 55, 132.905 , 488.0 );
    m_add_elements( "Gadolinium" , 64, 157.25  , 591.0 );
    m_add_elements( "Lutetium"   , 71, 174.97  , 694.0 );
    m_add_elements( "Tungsten"   , 74, 183.84  , 727.0 );
    m_add_elements( "Gold"       , 79, 196.967 , 790.0 );
    m_add_elements( "Thallium"   , 81, 204.37  , 810.0 );
    m_add_elements( "Lead" 	     , 82, 207.20  , 823.0 );
    m_add_elements( "Bismuth"    , 83, 208.98  , 823.0 );
    m_add_elements( "Uranium"    , 92, 238.03  , 890.0 );
}

//////////////////////////////////////////////////////////////////
//// Materials class /////////////////////////////////////////////
//////////////////////////////////////////////////////////////////

Materials::Materials() {} // // This class is used to build the material table

///:: Privates

// Check mandatory
bool Materials::m_check_mandatory() {

    if (m_nb_materials == 0) return false;
    else return true;
}

// Copy data to the GPU
void Materials::m_copy_materials_table_cpu2gpu() {

    ui32 n = m_nb_materials;
    ui32 k = m_nb_elements_total;

    // First allocate the GPU mem for the scene
    HANDLE_ERROR( cudaMalloc((void**) &data_d.nb_elements, n*sizeof(ui16)) );
    HANDLE_ERROR( cudaMalloc((void**) &data_d.index, n*sizeof(ui16)) );

    HANDLE_ERROR( cudaMalloc((void**) &data_d.mixture, k*sizeof(ui16)) );
    HANDLE_ERROR( cudaMalloc((void**) &data_d.atom_num_dens, k*sizeof(f32)) );

    HANDLE_ERROR( cudaMalloc((void**) &data_d.nb_atoms_per_vol, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &data_d.nb_electrons_per_vol, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &data_d.electron_mean_excitation_energy, n*sizeof(ui32)) );
    HANDLE_ERROR( cudaMalloc((void**) &data_d.rad_length, n*sizeof(f32)) );

    HANDLE_ERROR( cudaMalloc((void**) &data_d.photon_energy_cut, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &data_d.electron_energy_cut, n*sizeof(f32)) );

    HANDLE_ERROR( cudaMalloc((void**) &data_d.fX0, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &data_d.fX1, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &data_d.fD0, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &data_d.fC, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &data_d.fA, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &data_d.fM, n*sizeof(f32)) );

    HANDLE_ERROR( cudaMalloc((void**) &data_d.fF1, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &data_d.fF2, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &data_d.fEnergy0, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &data_d.fEnergy1, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &data_d.fEnergy2, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &data_d.fLogEnergy1, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &data_d.fLogEnergy2, n*sizeof(f32)) );
    HANDLE_ERROR( cudaMalloc((void**) &data_d.fLogMeanExcitationEnergy, n*sizeof(f32)) );

    HANDLE_ERROR( cudaMalloc((void**) &data_d.density, n*sizeof(f32)) );

    // Copy data to the GPU
    data_d.nb_materials = data_h.nb_materials;
    data_d.nb_elements_total = data_h.nb_elements_total;

    HANDLE_ERROR( cudaMemcpy( data_d.nb_elements, data_h.nb_elements,
                             n*sizeof(ui16), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( data_d.index, data_h.index,
                             n*sizeof(ui16), cudaMemcpyHostToDevice) );

    HANDLE_ERROR( cudaMemcpy( data_d.mixture, data_h.mixture,
                             k*sizeof(ui16), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( data_d.atom_num_dens, data_h.atom_num_dens,
                             k*sizeof(f32), cudaMemcpyHostToDevice) );

    HANDLE_ERROR( cudaMemcpy( data_d.nb_atoms_per_vol, data_h.nb_atoms_per_vol,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( data_d.nb_electrons_per_vol, data_h.nb_electrons_per_vol,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( data_d.electron_mean_excitation_energy, data_h.electron_mean_excitation_energy,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( data_d.rad_length, data_h.rad_length,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );

    HANDLE_ERROR( cudaMemcpy( data_d.photon_energy_cut, data_h.photon_energy_cut,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( data_d.electron_energy_cut, data_h.electron_energy_cut,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );

    HANDLE_ERROR( cudaMemcpy( data_d.fX0, data_h.fX0,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( data_d.fX1, data_h.fX1,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( data_d.fD0, data_h.fD0,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( data_d.fC, data_h.fC,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( data_d.fA, data_h.fA,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( data_d.fM, data_h.fM,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );

    HANDLE_ERROR( cudaMemcpy( data_d.fF1, data_h.fF1,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( data_d.fF2, data_h.fF2,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( data_d.fEnergy0, data_h.fEnergy0,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( data_d.fEnergy1, data_h.fEnergy1,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( data_d.fEnergy2, data_h.fEnergy2,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( data_d.fLogEnergy1, data_h.fLogEnergy1,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( data_d.fLogEnergy2, data_h.fLogEnergy2,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy( data_d.fLogMeanExcitationEnergy, data_h.fLogMeanExcitationEnergy,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );

    HANDLE_ERROR( cudaMemcpy( data_d.density, data_h.density,
                             n*sizeof(f32), cudaMemcpyHostToDevice) );

}

// Build the materials table according the list of materials
void Materials::m_build_materials_table(GlobalSimulationParameters params, std::vector<std::string> mats_list) {

    // First allocated data to the structure according the number of materials
    m_nb_materials = mats_list.size();
    data_h.nb_materials = mats_list.size();
    data_h.nb_elements = (ui16*)malloc(sizeof(ui16)*m_nb_materials);
    data_h.index = (ui16*)malloc(sizeof(ui16)*m_nb_materials);
    data_h.nb_atoms_per_vol = (f32*)malloc(sizeof(f32)*m_nb_materials);
    data_h.nb_electrons_per_vol = (f32*)malloc(sizeof(f32)*m_nb_materials);
    data_h.electron_mean_excitation_energy = (f32*)malloc(sizeof(f32)*m_nb_materials);
    data_h.rad_length = (f32*)malloc(sizeof(f32)*m_nb_materials);
    data_h.photon_energy_cut = (f32*)malloc(sizeof(f32)*m_nb_materials);
    data_h.electron_energy_cut = (f32*)malloc(sizeof(f32)*m_nb_materials);
    data_h.fX0 = (f32*)malloc(sizeof(f32)*m_nb_materials);
    data_h.fX1 = (f32*)malloc(sizeof(f32)*m_nb_materials);
    data_h.fD0 = (f32*)malloc(sizeof(f32)*m_nb_materials);
    data_h.fC = (f32*)malloc(sizeof(f32)*m_nb_materials);
    data_h.fA = (f32*)malloc(sizeof(f32)*m_nb_materials);
    data_h.fM = (f32*)malloc(sizeof(f32)*m_nb_materials);
    data_h.density = (f32*)malloc(sizeof(f32)*m_nb_materials);
    data_h.fF1 = (f32*)malloc(sizeof(f32)*m_nb_materials);
    data_h.fF2 = (f32*)malloc(sizeof(f32)*m_nb_materials);
    data_h.fEnergy0 = (f32*)malloc(sizeof(f32)*m_nb_materials);
    data_h.fEnergy1 = (f32*)malloc(sizeof(f32)*m_nb_materials);
    data_h.fEnergy2 = (f32*)malloc(sizeof(f32)*m_nb_materials);
    data_h.fLogEnergy1 = (f32*)malloc(sizeof(f32)*m_nb_materials);
    data_h.fLogEnergy2 = (f32*)malloc(sizeof(f32)*m_nb_materials);
    data_h.fLogMeanExcitationEnergy = (f32*)malloc(sizeof(f32)*m_nb_materials);

    i32 i, j;
    ui32 access_index = 0;
    ui32 fill_index = 0;
    std::string mat_name, elt_name;
    aMaterial cur_mat;

    i=0; while (i < m_nb_materials) {
        // get mat name
        mat_name = mats_list[i];

        // read mat from databse
        cur_mat = m_material_db.materials[mat_name];
        if (cur_mat.name == "") {
            printf("[ERROR] Material %s is not on your database (%s function)\n", mat_name.c_str(),__FUNCTION__);
            exit_simulation();
        }
        // get nb of elements
        data_h.nb_elements[i] = cur_mat.nb_elements;

        // compute index
        data_h.index[i] = access_index;
        access_index += cur_mat.nb_elements;

        ++i;
    }

    // nb of total elements
    m_nb_elements_total = access_index;
    data_h.nb_elements_total = access_index;
    data_h.mixture = (ui16*)malloc(sizeof(ui16)*access_index);
    data_h.atom_num_dens = (f32*)malloc(sizeof(f32)*access_index);

    // store mixture element and compute atomic density
    i=0; while (i < m_nb_materials) {

        // get mat name
        mat_name = mats_list[i];

        // read mat from database
        cur_mat = m_material_db.materials[mat_name];

        // get density
        data_h.density[i] = m_material_db.get_density( mat_name ) / gram;  // Why /gram ??  - JB

        data_h.nb_atoms_per_vol[i] = 0.0f;
        data_h.nb_electrons_per_vol[i] = 0.0f;

        j=0; while (j < m_material_db.get_nb_elements( mat_name )) {
            // read element name
            //elt_name = cur_mat.mixture_Z[j];
            elt_name = m_material_db.get_element_name( mat_name, j );

            // store Z
            data_h.mixture[fill_index] = m_material_db.get_element_Z( elt_name );

            // compute atom num dens (Avo*fraction*dens) / Az
            data_h.atom_num_dens[fill_index] = m_material_db.get_atom_num_dens( mat_name, j );

            // compute nb atoms per volume
            data_h.nb_atoms_per_vol[i] += data_h.atom_num_dens[fill_index];

            // compute nb electrons per volume
            data_h.nb_electrons_per_vol[i] += data_h.atom_num_dens[fill_index] *
                                              m_material_db.get_element_Z( elt_name );            

            ++j;
            ++fill_index;
        }

        /// electron Ionisation data
        m_material_db.compute_ioni_parameters( mat_name );

        data_h.electron_mean_excitation_energy[i] = m_material_db.get_mean_excitation();
        data_h.fLogMeanExcitationEnergy[i] = logf( m_material_db.get_mean_excitation() );

        // correction
        data_h.fX0[i] = m_material_db.get_X0_density();
        data_h.fX1[i] = m_material_db.get_X1_density();
        data_h.fD0[i] = m_material_db.get_D0_density();
        data_h.fC[i] = m_material_db.get_C_density();
        data_h.fA[i] = m_material_db.get_A_density();
        data_h.fM[i] = m_material_db.get_M_density();

        //eFluctuation parameters
        data_h.fF1[i] = m_material_db.get_F1_fluct();
        data_h.fF2[i] = m_material_db.get_F2_fluct();
        data_h.fEnergy0[i] = m_material_db.get_Energy0_fluct();
        data_h.fEnergy1[i] = m_material_db.get_Energy1_fluct();
        data_h.fEnergy2[i] = m_material_db.get_Energy2_fluct();
        data_h.fLogEnergy1[i] = m_material_db.get_LogEnergy1_fluct();
        data_h.fLogEnergy2[i] = m_material_db.get_LogEnergy2_fluct();

        /// others stuffs

        // cut (in energy for now, but should be change for range cut) TODO
        data_h.photon_energy_cut[i] = params.data_h.photon_cut;
        data_h.electron_energy_cut[i] = params.data_h.electron_cut;
        data_h.rad_length[i] = m_material_db.get_rad_len( mat_name );

        /// HERE compute and print range cut - DEBUGING - ALPHA
        //f32 Ecut = m_rangecut.convert_gamma(80*um, &data_h, i);
        f32 gEcut = m_rangecut.convert_gamma(100*um, data_h.mixture, data_h.nb_elements[i], data_h.atom_num_dens, data_h.index[i]);
        f32 eEcut = m_rangecut.convert_electron(100.0*um, data_h.mixture, data_h.nb_elements[i], data_h.atom_num_dens, data_h.density[i], data_h.index[i]);


        if ( params.data_h.display_energy_cuts )
        {
            printf("[GGEMS] Range cut    material: %s       gamma: %f keV\n", mat_name.c_str(), gEcut/keV);
            printf("[GGEMS] Range cut    material: %s    electron: %f keV\n", mat_name.c_str(), eEcut/keV);
            printf("\n");
        }

        ++i;
    }

}


///:: Mains

// Load default materials database (wrapper to the class MaterialDataBase)
void Materials::load_materials_database() {
    std::string filename = std::string(getenv("GGEMSHOME"));
    filename += "/data/mats.dat";
    m_material_db.load_materials(filename);
}

// Load materials database from a given file (wrapper to the class MaterialDataBase)
void Materials::load_materials_database(std::string filename) {
    m_material_db.load_materials(filename);
}

/*
// Load elements database from a given file (wrapper to the class MaterialDataBase)
void Materials::load_elements_database(std::string filename) {
    m_material_db.load_elements(filename);
}
*/



//// Table containing every definition of the materials used in the world
//struct MaterialsTable {
//    ui32 nb_materials;              // n
//    ui32 nb_elements_total;         // k

//    ui16 *nb_elements;        // n
//    ui16 *index;              // n

//    ui16 *mixture;            // k
//    f32 *atom_num_dens;                   // k

//    f32 *nb_atoms_per_vol;                // n
//    f32 *nb_electrons_per_vol;            // n
//    f32 *electron_mean_excitation_energy; // n
//    f32 *rad_length;                      // n

//    // Cut
//    f32 *photon_energy_cut;               // n
//    f32 *electron_energy_cut;             // n

//    //parameters of the density correction
//    f32 *fX0;                             // n
//    f32 *fX1;
//    f32 *fD0;
//    f32 *fC;
//    f32 *fA;
//    f32 *fM;

//  // parameters of the energy loss fluctuation model:
//    f32 *fF1;
//    f32 *fF2;
//    f32 *fEnergy0;
//    f32 *fEnergy1;
//    f32 *fEnergy2;
//    f32 *fLogEnergy1;
//    f32 *fLogEnergy2;
//    f32 *fLogMeanExcitationEnergy;

//    f32 *density;
//};


// Print Material table
void Materials::print()
{
    printf("[GGEMS] Materials table:\n");
    printf("[GGEMS]    Nb materials: %i\n", data_h.nb_materials);
    printf("[GGEMS]    Nb total elements: %i\n", data_h.nb_elements_total);
    printf("[GGEMS]\n");

    ui32 mat_id = 0; while ( mat_id < data_h.nb_materials )
    {
        ui32 index = data_h.index[ mat_id ];
        printf("[GGEMS]    %s\n", m_materials_list_name[ mat_id ].c_str());
        printf("[GGEMS]       Nb elements: %i\n",  data_h.nb_elements[ mat_id ]);
        printf("[GGEMS]       Access index: %i\n",  index);
        printf("[GGEMS]       Nb atoms per vol: %e\n",  data_h.nb_atoms_per_vol[ mat_id ]);
        printf("[GGEMS]       Nb electrons per vol: %e\n", data_h.nb_electrons_per_vol[ mat_id ]);
        printf("[GGEMS]       Electron mean exitation energy: %e\n", data_h.electron_mean_excitation_energy[ mat_id ]);
        printf("[GGEMS]       Rad length: %e\n", data_h.rad_length[ mat_id ]);
        printf("[GGEMS]       Density: %e\n", data_h.density[ mat_id ]);
        printf("[GGEMS]       Photon energy cut: %e\n", data_h.photon_energy_cut[ mat_id ]);
        printf("[GGEMS]       Electon energy cut: %e\n", data_h.electron_energy_cut[ mat_id ]);
        printf("[GGEMS]\n");
        printf("[GGEMS]       Density correction:\n");
        printf("[GGEMS]          fX0: %e\n", data_h.fX0[ mat_id ]);
        printf("[GGEMS]          fX1: %e\n", data_h.fX1[ mat_id ]);
        printf("[GGEMS]          fD0: %e\n", data_h.fD0[ mat_id ]);
        printf("[GGEMS]          fC: %e\n", data_h.fC[ mat_id ]);
        printf("[GGEMS]          fA: %e\n", data_h.fA[ mat_id ]);
        printf("[GGEMS]          fM: %e\n", data_h.fM[ mat_id ]);
        printf("[GGEMS]\n");
        printf("[GGEMS]       Energy loss fluctuation:\n");
        printf("[GGEMS]          fF1: %e\n", data_h.fF1[ mat_id ]);
        printf("[GGEMS]          fF2: %e\n", data_h.fF2[ mat_id ]);
        printf("[GGEMS]          fEnergy0: %e\n", data_h.fEnergy0[ mat_id ]);
        printf("[GGEMS]          fEnergy1: %e\n", data_h.fEnergy1[ mat_id ]);
        printf("[GGEMS]          fEnergy2: %e\n", data_h.fEnergy2[ mat_id ]);
        printf("[GGEMS]          fLogEnergy1: %e\n", data_h.fLogEnergy1[ mat_id ]);
        printf("[GGEMS]          fLogEnergy2: %e\n", data_h.fLogEnergy2[ mat_id ]);
        printf("[GGEMS]          fLogMeanExcitationEnergy: %e\n", data_h.fLogMeanExcitationEnergy[ mat_id ]);
        printf("[GGEMS]\n");
        printf("[GGEMS]       Mixture:\n");

        ui32 elt_id = 0; while ( elt_id < data_h.nb_elements[ mat_id ] )
        {
            printf("[GGEMS]          Z: %i   Atom Num Dens: %e\n", data_h.mixture[ index+elt_id ], data_h.atom_num_dens[ index+elt_id ]);
            ++elt_id;
        }
        printf("[GGEMS]\n");
        ++mat_id;
    }


}

/*
// Add materials to the main list and update the corresponding indices
void MaterialManager::add_materials_and_update_indices(std::vector<std::string> mats_list, ui16 *data, ui32 ndata) {


    ui16 local_id_mat=0; while (local_id_mat<mats_list.size()) {

        ui16 glb_id_mat = m_get_material_index(mats_list[local_id_mat]);

        // If the material index is different to the mats_list index,
        // the object local material list have to be
        // re-index considering the main (and global) material list
        if (glb_id_mat != local_id_mat) {

            ui32 i=0; while (i<ndata) {
                if (data[i] == local_id_mat) {
                    data[i] = glb_id_mat;
                }
                ++i;
            }

        }

        ++local_id_mat;
    }


}
*/


//// Build the materials table according the object contains in the world
//void MaterialManager::free_materials_table() {


//    free(materials_table.nb_elements);
//    free(materials_table.index);
//    free(materials_table.nb_atoms_per_vol);
//    free(materials_table.nb_electrons_per_vol);
//    free(materials_table.electron_mean_excitation_energy);
//    free(materials_table.rad_length);
//    free(materials_table.fX0);
//    free(materials_table.fX1);
//    free(materials_table.fD0);
//    free(materials_table.fC);
//    free(materials_table.fA);
//    free(materials_table.fM);
//    free(materials_table.density);
//    free(materials_table.fF1);
//    free(materials_table.fF2);
//    free(materials_table.fEnergy0);
//    free(materials_table.fEnergy1);
//    free(materials_table.fEnergy2);
//    free(materials_table.fLogEnergy1);
//    free(materials_table.fLogEnergy2);
//    free(materials_table.fLogMeanExcitationEnergy);
//    free(materials_table.mixture);
//    free(materials_table.atom_num_dens);


//    //delete mat_table_h;

//}



// Init
void Materials::initialize(std::vector<std::string> mats_list, GlobalSimulationParameters params) {

    m_nb_materials = mats_list.size();
    m_materials_list_name = mats_list;

    // Check if everything was set properly
    if ( !m_check_mandatory() ) {
        print_error("Missing materials definition!");
        exit_simulation();
    }

    // Load elements data base
    m_material_db.load_elements();
    //load_elements_database( "data/materials/elts.dat" );

    // Build materials table
    m_build_materials_table(params, mats_list);
    
    // Copy data to the GPU
    if (params.data_h.device_target == GPU_DEVICE) m_copy_materials_table_cpu2gpu();
}

#endif
