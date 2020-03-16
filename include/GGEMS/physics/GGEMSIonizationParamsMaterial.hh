#ifndef GUARD_GGEMS_PHYSICS_GGEMSIONIZATIONPARAMSMATERIAL_HH
#define GUARD_GGEMS_PHYSICS_GGEMSIONIZATIONPARAMSMATERIAL_HH

/*!
  \file GGEMSIonizationParamsMaterial.hh

  \brief GGEMS class managing some physical params for ionization process for material

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 10, 2020
*/

#include <memory>

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/physics/GGEMSMaterialsManager.hh"

/*!
  \namespace GGEMSDensityParams
  \brief namespace data for density correction
*/
namespace GGEMSDensityParams
{
  inline static constexpr GGfloat data[96][9] = {
    // Eplasma, rho, -C, X_0, X_1, a, m, delta_0, delta_max
    { 0.000f, 0.000f,  0.0000f,  0.0000f, 0.0000f, 0.000000f, 0.0000f, 0.00f, 0.000f},  // Z=0
    { 0.263f, 1.412f,  9.5835f,  1.8639f, 3.2718f, 0.140920f, 5.7273f, 0.00f, 0.024f},
    { 0.263f, 1.700f, 11.1393f,  2.2017f, 3.6122f, 0.134430f, 5.8347f, 0.00f, 0.024f},
    {13.844f, 1.535f,  3.1221f,  0.1304f, 1.6397f, 0.951360f, 2.4993f, 0.14f, 0.062f},
    {26.096f, 1.908f,  2.7847f,  0.0392f, 1.6922f, 0.803920f, 2.4339f, 0.14f, 0.029f},
    {30.170f, 2.320f,  2.8477f,  0.0305f, 1.9688f, 0.562240f, 2.4512f, 0.14f, 0.024f},
    {28.803f, 2.376f,  2.9925f, -0.0351f, 2.4860f, 0.202400f, 3.0036f, 0.10f, 0.038f},
    { 0.695f, 1.984f, 10.5400f,  1.7378f, 4.1323f, 0.153490f, 3.2125f, 0.00f, 0.086f},
    { 0.744f, 2.314f, 10.7004f,  1.7541f, 4.3213f, 0.117780f, 3.2913f, 0.00f, 0.101f},
    { 0.788f, 2.450f, 10.9653f,  1.8433f, 4.4096f, 0.110830f, 3.2962f, 0.00f, 0.121f},
    { 0.587f, 2.577f, 11.9041f,  2.0735f, 4.6421f, 0.080640f, 3.5771f, 0.00f, 0.110f},
    {19.641f, 2.648f,  5.0526f,  0.2880f, 3.1962f, 0.077720f, 3.6452f, 0.08f, 0.098f},
    {26.708f, 2.331f,  4.5297f,  0.1499f, 3.0668f, 0.081163f, 3.6166f, 0.08f, 0.073f},
    {32.860f, 2.180f,  4.2395f,  0.1708f, 3.0127f, 0.080240f, 3.6345f, 0.12f, 0.061f},
    {31.055f, 2.103f,  4.4351f,  0.2014f, 2.8715f, 0.149210f, 3.2546f, 0.14f, 0.059f},
    {29.743f, 2.056f,  4.5214f,  0.1696f, 2.7815f, 0.236100f, 2.9158f, 0.14f, 0.057f},
    {28.789f, 2.131f,  4.6659f,  0.1580f, 2.7159f, 0.339920f, 2.6456f, 0.14f, 0.059f},
    { 1.092f, 1.734f, 11.1421f,  1.5555f, 4.2994f, 0.198490f, 2.9702f, 0.00f, 0.041f},
    { 0.789f, 1.753f, 11.9480f,  1.7635f, 4.4855f, 0.197140f, 2.9618f, 0.00f, 0.037f},
    {18.650f, 1.830f,  5.6423f,  0.3851f, 3.1724f, 0.198270f, 2.9233f, 0.10f, 0.035f},
    {25.342f, 1.666f,  5.0396f,  0.3228f, 3.1191f, 0.156430f, 3.0745f, 0.14f, 0.031f},
    {34.050f, 1.826f,  4.6949f,  0.1640f, 3.0593f, 0.157540f, 3.0517f, 0.10f, 0.027f},
    {41.619f, 1.969f,  4.4450f,  0.0957f, 3.0386f, 0.156620f, 3.0302f, 0.12f, 0.025f},
    {47.861f, 2.070f,  4.2659f,  0.0691f, 3.0322f, 0.154360f, 3.0163f, 0.14f, 0.024f},
    {52.458f, 2.181f,  4.1781f,  0.0340f, 3.0451f, 0.154190f, 2.9896f, 0.14f, 0.023f},
    {53.022f, 2.347f,  4.2702f,  0.0447f, 3.1074f, 0.149730f, 2.9796f, 0.14f, 0.021f},
    {55.172f, 2.504f,  4.2911f, -0.0012f, 3.1531f, 0.146000f, 2.9632f, 0.12f, 0.021f},
    {58.188f, 2.626f,  4.2601f, -0.0187f, 3.1790f, 0.144740f, 2.9502f, 0.12f, 0.019f},
    {59.385f, 2.889f,  4.3115f, -0.0566f, 3.1851f, 0.164960f, 2.8430f, 0.10f, 0.020f},
    {58.270f, 2.956f,  4.4190f, -0.0254f, 3.2792f, 0.143390f, 2.9044f, 0.08f, 0.019f},
    {52.132f, 3.142f,  4.6906f,  0.0049f, 3.3668f, 0.147140f, 2.8652f, 0.08f, 0.019f},
    {46.688f, 2.747f,  4.9353f,  0.2267f, 3.5434f, 0.094400f, 3.1314f, 0.14f, 0.019f},
    {44.141f, 2.461f,  5.1411f,  0.3376f, 3.6096f, 0.071880f, 3.3306f, 0.14f, 0.025f},
    {45.779f, 2.219f,  5.0510f,  0.1767f, 3.5702f, 0.066330f, 3.4176f, 0.00f, 0.030f},
    {40.112f, 2.104f,  5.3210f,  0.2258f, 3.6264f, 0.065680f, 3.4317f, 0.10f, 0.024f},
    { 1.604f, 1.845f, 11.7307f,  1.5262f, 4.9899f, 0.063350f, 3.4670f, 0.00f, 0.022f},
    { 1.114f, 1.770f, 12.5115f,  1.7158f, 5.0748f, 0.074460f, 3.4051f, 0.00f, 0.025f},
    {23.467f, 1.823f,  6.4776f,  0.5737f, 3.7995f, 0.072610f, 3.4177f, 0.14f, 0.026f},
    {30.244f, 1.707f,  5.9867f,  0.4585f, 3.6778f, 0.071650f, 3.4435f, 0.14f, 0.026f},
    {40.346f, 1.649f,  5.4801f,  0.3608f, 3.5542f, 0.071380f, 3.4565f, 0.14f, 0.027f},
    {48.671f, 1.638f,  5.1774f,  0.2957f, 3.4890f, 0.071770f, 3.4533f, 0.14f, 0.028f},
    {56.039f, 1.734f,  5.0141f,  0.1785f, 3.2201f, 0.138830f, 3.0930f, 0.14f, 0.036f},
    {60.951f, 1.658f,  4.8793f,  0.2267f, 3.2784f, 0.105250f, 3.2549f, 0.14f, 0.030f},
    {64.760f, 1.727f,  4.7769f,  0.0949f, 3.1253f, 0.165720f, 2.9738f, 0.14f, 0.040f},
    {66.978f, 1.780f,  4.7694f,  0.0599f, 3.0834f, 0.193420f, 2.8707f, 0.14f, 0.046f},
    {67.128f, 1.804f,  4.8008f,  0.0576f, 3.1069f, 0.192050f, 2.8633f, 0.14f, 0.046f},
    {65.683f, 1.911f,  4.9358f,  0.0563f, 3.0555f, 0.241780f, 2.7239f, 0.14f, 0.047f},
    {61.635f, 1.933f,  5.0630f,  0.0657f, 3.1074f, 0.245850f, 2.6899f, 0.14f, 0.052f},
    {55.381f, 1.895f,  5.2727f,  0.1281f, 3.1667f, 0.246090f, 2.6772f, 0.14f, 0.051f},
    {50.896f, 1.851f,  5.5211f,  0.2406f, 3.2032f, 0.238790f, 2.7144f, 0.14f, 0.044f},
    {50.567f, 1.732f,  5.5340f,  0.2879f, 3.2959f, 0.186890f, 2.8576f, 0.14f, 0.037f},
    {48.242f, 1.645f,  5.6241f,  0.3189f, 3.3489f, 0.166520f, 2.9319f, 0.14f, 0.034f},
    {45.952f, 1.577f,  5.7131f,  0.3296f, 3.4418f, 0.138150f, 3.0354f, 0.14f, 0.033f},
    {41.348f, 1.498f,  5.9488f,  0.0549f, 3.2596f, 0.237660f, 2.7276f, 0.00f, 0.045f},
    { 1.369f, 1.435f, 12.7281f,  1.5630f, 4.7371f, 0.233140f, 2.7414f, 0.00f, 0.043f},
    {25.370f, 1.462f,  6.9135f,  0.5473f, 3.5914f, 0.182330f, 2.8866f, 0.14f, 0.035f},
    {34.425f, 1.410f,  6.3153f,  0.4190f, 3.4547f, 0.182680f, 2.8906f, 0.14f, 0.035f},
    {45.792f, 1.392f,  5.7850f,  0.3161f, 3.3293f, 0.185910f, 2.8828f, 0.14f, 0.036f},
    {47.834f, 1.461f,  5.7837f,  0.2713f, 3.3432f, 0.188850f, 2.8592f, 0.14f, 0.040f},
    {48.301f, 1.520f,  5.8096f,  0.2333f, 3.2773f, 0.232650f, 2.7331f, 0.14f, 0.041f},
    {48.819f, 1.588f,  5.8290f,  0.1984f, 3.3063f, 0.235300f, 2.7050f, 0.14f, 0.044f},
    {50.236f, 1.672f,  5.8224f,  0.1627f, 3.3199f, 0.242800f, 2.6674f, 0.14f, 0.048f},
    {50.540f, 1.749f,  5.8597f,  0.1520f, 3.3460f, 0.246980f, 2.6403f, 0.14f, 0.053f},
    {42.484f, 1.838f,  6.2278f,  0.1888f, 3.4633f, 0.244480f, 2.6245f, 0.14f, 0.060f},
    {51.672f, 1.882f,  5.8738f,  0.1058f, 3.3932f, 0.251090f, 2.5977f, 0.14f, 0.061f},
    {52.865f, 1.993f,  5.9045f,  0.0947f, 3.4224f, 0.244530f, 2.6056f, 0.14f, 0.063f},
    {53.698f, 2.081f,  5.9183f,  0.0822f, 3.4474f, 0.246650f, 2.5849f, 0.14f, 0.061f},
    {54.467f, 2.197f,  5.9587f,  0.0761f, 3.4782f, 0.246380f, 2.5726f, 0.14f, 0.062f},
    {55.322f, 2.260f,  5.9521f,  0.0648f, 3.4922f, 0.248230f, 2.5573f, 0.14f, 0.061f},
    {56.225f, 2.333f,  5.9677f,  0.0812f, 3.5085f, 0.241890f, 2.5469f, 0.14f, 0.062f},
    {47.546f, 2.505f,  6.3325f,  0.1199f, 3.6246f, 0.252950f, 2.5141f, 0.14f, 0.071f},
    {57.581f, 2.348f,  5.9785f,  0.1560f, 3.5218f, 0.240330f, 2.5643f, 0.14f, 0.054f},
    {66.770f, 2.174f,  5.7139f,  0.1965f, 3.4337f, 0.229180f, 2.6155f, 0.14f, 0.035f},
    {74.692f, 2.070f,  5.5262f,  0.2117f, 3.4805f, 0.177980f, 2.7623f, 0.14f, 0.030f},
    {80.315f, 1.997f,  5.4059f,  0.2167f, 3.4960f, 0.155090f, 2.8447f, 0.14f, 0.027f},
    {83.846f, 1.976f,  5.3445f,  0.0559f, 3.4845f, 0.151840f, 2.8627f, 0.08f, 0.026f},
    {86.537f, 1.947f,  5.3083f,  0.0891f, 3.5414f, 0.127510f, 2.9608f, 0.10f, 0.023f},
    {86.357f, 1.927f,  5.3418f,  0.0819f, 3.5480f, 0.126900f, 2.9658f, 0.10f, 0.023f},
    {84.389f, 1.965f,  5.4732f,  0.1484f, 3.6212f, 0.111280f, 3.0417f, 0.12f, 0.021f},
    {80.215f, 1.926f,  5.5747f,  0.2021f, 3.6979f, 0.097560f, 3.1101f, 0.14f, 0.020f},
    {66.977f, 1.904f,  5.9605f,  0.2756f, 3.7275f, 0.110140f, 3.0519f, 0.14f, 0.021f},
    {62.104f, 1.814f,  6.1365f,  0.3491f, 3.8044f, 0.094550f, 3.1450f, 0.14f, 0.019f},
    {61.072f, 1.755f,  6.2018f,  0.3776f, 3.8073f, 0.093590f, 3.1608f, 0.14f, 0.019f},
    {56.696f, 1.684f,  6.3505f,  0.4152f, 3.8248f, 0.094100f, 3.1671f, 0.14f, 0.020f},
    {55.773f, 1.637f,  6.4003f,  0.4267f, 3.8293f, 0.092820f, 3.1830f, 0.14f, 0.020f},
    { 1.708f, 1.458f, 13.2839f,  1.5368f, 4.9889f, 0.207980f, 2.7409f, 0.00f, 0.057f},
    {40.205f, 1.403f,  7.0452f,  0.5991f, 3.9428f, 0.088040f, 3.2454f, 0.14f, 0.022f},
    {57.254f, 1.380f,  6.3742f,  0.4559f, 3.7966f, 0.085670f, 3.2683f, 0.14f, 0.023f},
    {61.438f, 1.363f,  6.2473f,  0.4202f, 3.7681f, 0.086550f, 3.2610f, 0.14f, 0.025f},
    {70.901f, 1.420f,  6.0327f,  0.3144f, 3.5079f, 0.147700f, 2.9845f, 0.14f, 0.036f},
    {77.986f, 1.447f,  5.8694f,  0.2260f, 3.3721f, 0.196770f, 2.8171f, 0.14f, 0.043f},
    {81.221f, 1.468f,  5.8149f,  0.1869f, 3.3690f, 0.197410f, 2.8082f, 0.14f, 0.043f},
    {80.486f, 1.519f,  5.8748f,  0.1557f, 3.3981f, 0.204190f, 2.7679f, 0.14f, 0.057f},
    {66.607f, 1.552f,  6.2813f,  0.2274f, 3.5021f, 0.203080f, 2.7615f, 0.14f, 0.056f},
    {66.022f, 1.559f,  6.3097f,  0.2484f, 3.5160f, 0.202570f, 2.7579f, 0.14f, 0.056f},
    {67.557f, 1.574f,  6.2912f,  0.2378f, 3.5186f, 0.201920f, 2.7560f, 0.14f, 0.062f}
  };
}

/*!
  \class GGEMSIonizationParamsMaterial
  \brief GGEMS class handling material(s) for a specific navigator
*/
class GGEMS_EXPORT GGEMSIonizationParamsMaterial
{
  public:
    /*!
      \brief GGEMSIonizationParamsMaterial constructor
    */
    explicit GGEMSIonizationParamsMaterial(GGEMSSingleMaterial const* material);

    /*!
      \brief GGEMSIonizationParamsMaterial destructor
    */
    ~GGEMSIonizationParamsMaterial(void);

    /*!
      \fn GGEMSIonizationParamsMaterial(GGEMSIonizationParamsMaterial const& ionization_params) = delete
      \param ionization params - reference on the GGEMS ionization params
      \brief Avoid copy by reference
    */
    GGEMSIonizationParamsMaterial(GGEMSIonizationParamsMaterial const& ionization_params) = delete;

    /*!
      \fn GGEMSIonizationParamsMaterial& operator=(GGEMSIonizationParamsMaterial const& ionization params) = delete
      \param ionization params - reference on the GGEMS ionization params
      \brief Avoid assignement by reference
    */
    GGEMSIonizationParamsMaterial& operator=(GGEMSIonizationParamsMaterial const& ionization_params) = delete;

    /*!
      \fn GGEMSIonizationParamsMaterial(GGEMSIonizationParamsMaterial const&& ionization_params) = delete
      \param ionization params - rvalue reference on the GGEMS ionization params
      \brief Avoid copy by rvalue reference
    */
    GGEMSIonizationParamsMaterial(GGEMSIonizationParamsMaterial const&& ionization_params) = delete;

    /*!
      \fn GGEMSIonizationParamsMaterial& operator=(GGEMSIonizationParamsMaterial const&& ionization params) = delete
      \param ionization params - rvalue reference on the GGEMS ionization params
      \brief Avoid copy by rvalue reference
    */
    GGEMSIonizationParamsMaterial& operator=(GGEMSIonizationParamsMaterial const&& ionization_params) = delete;

    /*!
      \fn inline GGfloat GetMeanExcitationEnergy(void) const
      \return mean excitation energy
      \brief get the mean excitation energy
    */
    inline GGfloat GetMeanExcitationEnergy(void) const {return mean_excitation_energy_;}

    /*!
      \fn inline GGfloat GetLogMeanExcitationEnergy(void) const
      \return log mean excitation energy
      \brief get the log mean excitation energy
    */
    inline GGfloat GetLogMeanExcitationEnergy(void) const {return log_mean_excitation_energy_;}

    /*!
      \fn inline GGfloat GetRadiationLength(void) const
      \return radiation length
      \brief get the radiation length
    */
    inline GGfloat GetRadiationLength(void) const {return radiation_length_;}

    /*!
      \fn inline GGfloat GetX0Density(void) const
      \return x0 density
      \brief get the x0 density
    */
    inline GGfloat GetX0Density(void) const {return x0_density_;}

    /*!
      \fn inline GGfloat GetX1Density(void) const
      \return x1 density
      \brief get the x1 density
    */
    inline GGfloat GetX1Density(void) const {return x1_density_;}

    /*!
      \fn inline GGfloat GetD0Density(void) const
      \return d0 density
      \brief get the d0 density
    */
    inline GGfloat GetD0Density(void) const {return d0_density_;}

    /*!
      \fn inline GGfloat GetCDensity(void) const
      \return c density
      \brief get the c density
    */
    inline GGfloat GetCDensity(void) const {return c_density_;}

    /*!
      \fn inline GGfloat GetADensity(void) const
      \return a density
      \brief get the a density
    */
    inline GGfloat GetADensity(void) const {return a_density_;}

    /*!
      \fn inline GGfloat GetMDensity(void) const
      \return m density
      \brief get the m density
    */
    inline GGfloat GetMDensity(void) const {return m_density_;}

    /*!
      \fn inline GGfloat GetF1Fluct(void) const
      \return f1 fluctuation
      \brief get the f1 fluctuation
    */
    inline GGfloat GetF1Fluct(void) const {return f1_fluct_;}

    /*!
      \fn inline GGfloat GetF2Fluct(void) const
      \return f2 fluctuation
      \brief get the f2 fluctuation
    */
    inline GGfloat GetF2Fluct(void) const {return f2_fluct_;}

    /*!
      \fn inline GGfloat GetEnergy0Fluct(void) const
      \return energy 0 fluctuation
      \brief get the energy 0 fluctuation
    */
    inline GGfloat GetEnergy0Fluct(void) const {return energy0_fluct_;}

    /*!
      \fn inline GGfloat GetEnergy1Fluct(void) const
      \return energy 1 fluctuation
      \brief get the energy 1 fluctuation
    */
    inline GGfloat GetEnergy1Fluct(void) const {return energy1_fluct_;}

    /*!
      \fn inline GGfloat GetEnergy2Fluct(void) const
      \return energy 2 fluctuation
      \brief get the energy 2 fluctuation
    */
    inline GGfloat GetEnergy2Fluct(void) const {return energy2_fluct_;}

    /*!
      \fn inline GGfloat GetLogEnergy1Fluct(void) const
      \return log energy 1 fluctuation
      \brief get the log energy 1 fluctuation
    */
    inline GGfloat GetLogEnergy1Fluct(void) const {return log_energy1_fluct_;}

    /*!
      \fn inline GGfloat GetLogEnergy2Fluct(void) const
      \return log energy 2 fluctuation
      \brief get the log energy 2 fluctuation
    */
    inline GGfloat GetLogEnergy2Fluct(void) const {return log_energy2_fluct_;}

  private:
    /*!
      \fn void ComputeIonizationParameters(void)
      \brief Computing all ionization parameters for a material
    */
    void ComputeIonizationParameters(void);

  private:
    // parameters for mean energy loss calculation
    GGfloat mean_excitation_energy_; /*!< Mean excitation energy */
    GGfloat log_mean_excitation_energy_; /*!< Log of mean excitation energy */
    GGfloat radiation_length_; /*!< Radiation length */

    // parameters of the density correction
    GGfloat x0_density_; /*!< x0 param for density correction */
    GGfloat x1_density_; /*!< x1 param for density correction */
    GGfloat d0_density_; /*!< d0 param for density correction */
    GGfloat c_density_; /*!< c param for density correction */
    GGfloat a_density_; /*!< a param for density correction */
    GGfloat m_density_; /*!< m param for density correction */

    // parameters of the energy loss fluctuation model
    GGfloat f1_fluct_; /*!< f1 param for energy loss fluctuation model */
    GGfloat f2_fluct_; /*!< f2 param for energy loss fluctuation model */
    GGfloat energy0_fluct_; /*!< energy 0 param for energy loss fluctuation model */
    GGfloat energy1_fluct_; /*!< energy 1 param for energy loss fluctuation model */
    GGfloat energy2_fluct_; /*!< energy 2 param for energy loss fluctuation model */
    GGfloat log_energy1_fluct_; /*!< log energy 1 param for energy loss fluctuation model */
    GGfloat log_energy2_fluct_; /*!< log energy 2 param for energy loss fluctuation model */

    GGEMSSingleMaterial const* material_; /*!< Pointer to a material */
};

#endif // End of GUARD_GGEMS_PHYSICS_GGEMSIONIZATIONPARAMSMATERIAL_HH
