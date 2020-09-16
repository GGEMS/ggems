#ifndef GUARD_GGEMS_PHYSICS_GGEMSRAYLEIGHSCATTERINGMODELS_HH
#define GUARD_GGEMS_PHYSICS_GGEMSRAYLEIGHSCATTERINGMODELS_HH

/*!
  \file GGEMSRayleighScatteringModels.hh

  \brief Models for Rayleigh scattering, only for OpenCL kernel usage

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday September 14, 2020
*/

#ifdef OPENCL_COMPILER

__constant GGfloat FACTOR = 32526509815670243328.0f; // 0.5*HC*HC -> HC = cm/(H_PLANCK*C_LIGHT)

__constant GGfloat PP0[101] = { 0.0f, 
  0.0f, 2.0f, 5.21459f, 10.2817f, 3.66207f, 3.63903f, 3.71155f, 36.5165f, 3.43548f, 3.40045f,     // 1-10 
  2.87811f, 3.35541f, 3.21141f, 2.95234f, 3.02524f, 126.146f, 175.044f, 162.0f, 296.833f, 300.994f,     // 11-20 
  373.186f, 397.823f, 430.071f, 483.293f, 2.14885f, 335.553f, 505.422f, 644.739f, 737.017f, 707.575f,     // 21-30 
  3.8094f, 505.957f, 4.10347f, 574.665f, 15.5277f, 10.0991f, 4.95013f, 16.3391f, 6.20836f, 3.52767f,     // 31-40 
  2.7763f, 2.19565f, 12.2802f, 965.741f, 1011.09f, 2.85583f, 3.65673f, 225.777f, 1.95284f, 15.775f,     // 41-50 
  39.9006f, 3.7927f, 64.7339f, 1323.91f, 3.73723f, 2404.54f, 28.3408f, 29.9869f, 217.128f, 71.7138f,     // 51-60 
  255.42f, 134.495f, 3364.59f, 425.326f, 449.405f, 184.046f, 3109.04f, 193.133f, 3608.48f, 152.967f,     // 61-70 
  484.517f, 422.591f, 423.518f, 393.404f, 437.172f, 432.356f, 478.71f, 455.097f, 495.237f, 417.8f,     // 71-80 
  3367.95f, 3281.71f, 3612.56f, 3368.73f, 3407.46f, 40.2866f, 641.24f, 826.44f, 3579.13f, 4916.44f,     // 81-90 
  930.184f, 887.945f, 3490.96f, 4058.6f, 3068.1f, 3898.32f, 1398.34f, 5285.18f, 1, 872.368f     // 91-100 
};

__constant GGfloat PP1[101] = { 0.0f, 
  1.f, 2.f, 3.7724f, 2.17924f, 11.9967f, 17.7772f, 23.5265f, 23.797f, 39.9937f, 46.7748f,     // 1-10 
  60.0f, 68.6446f, 81.7887f, 98.0f, 112.0f, 128.0f, 96.7939f, 162.0f, 61.5575f, 96.4218f,     // 11-20 
  65.4084f, 83.3079f, 96.2889f, 90.123f, 312.0f, 338.0f, 181.943f, 94.3868f, 54.5084f, 132.819f,     // 21-30 
  480.0f, 512.0f, 544.0f, 578.0f, 597.472f, 647.993f, 682.009f, 722.0f, 754.885f, 799.974f,     // 31-40 
  840.0f, 882.0f, 924.0f, 968.0f, 1012.0f, 1058.0f, 1104.0f, 1151.95f, 1199.05f, 1250.0f,     // 41-50 
  1300.0f, 1352.0f, 1404.0f, 1458.0f, 1512.0f, 729.852f, 1596.66f, 1682.0f, 1740.0f, 1800.0f,     // 51-60 
  1605.79f, 1787.51f, 603.151f, 2048.0f, 2112.0f, 1993.95f, 334.907f, 2312.0f, 885.149f, 2337.19f,     // 61-70 
  2036.48f, 2169.41f, 2241.49f, 2344.6f, 2812.0f, 2888.0f, 2964.0f, 2918.04f, 2882.97f, 2938.74f,     // 71-80 
  2716.13f, 511.66f, 581.475f, 594.305f, 672.232f, 3657.71f, 3143.76f, 3045.56f, 3666.7f, 1597.84f,     // 81-90 
  3428.87f, 3681.22f, 1143.31f, 1647.17f, 1444.9f, 1894.33f, 3309.12f, 2338.59f, 4900.0f, 4856.61f     // 91-100 
};

__constant GGfloat PP2[101] = { 0.0f, 
  0.0f, 0.0f, 0.0130091f, 3.53906f, 9.34125f, 14.5838f, 21.7619f, 3.68644f, 37.5709f, 49.8248f,     // 1-10 
  58.1219f, 72.0f, 83.9999f, 95.0477f, 109.975f, 1.85351f, 17.1623f, 0.0f, 2.60927f, 2.58422f,     // 11-20 
  2.4053f, 2.86948f, 2.63999f, 2.58417f, 310.851f, 2.44683f, 41.6348f, 44.8739f, 49.4746f, 59.6053f,     // 21-30 
  477.191f, 6.04261f, 540.897f, 3.33531f, 612.0f, 637.908f, 682.041f, 705.661f, 759.906f, 796.498f,     // 31-40 
  838.224f, 879.804f, 912.72f, 2.25892f, 1.90993f, 1055.14f, 1101.34f, 926.275f, 1200.0f, 1234.23f,     // 41-50 
  1261.1f, 1348.21f, 1340.27f, 134.085f, 1509.26f, 1.60851f, 1624.0f, 1652.01f, 1523.87f, 1728.29f,     // 51-60 
  1859.79f, 1922.0f, 1.25916f, 1622.67f, 1663.6f, 2178.0f, 1045.05f, 2118.87f, 267.371f, 2409.84f,     // 61-70 
  2520.0f, 2592.0f, 2664.0f, 2738.0f, 2375.83f, 2455.64f, 2486.29f, 2710.86f, 2862.79f, 3043.46f,     // 71-80 
  476.925f, 2930.63f, 2694.96f, 3092.96f, 3145.31f, 3698.0f, 3784.0f, 3872.0f, 675.166f, 1585.71f,     // 81-90 
  3921.95f, 3894.83f, 4014.73f, 3130.23f, 4512.0f, 3423.35f, 4701.53f, 1980.23f, 4900.0f, 4271.02f     // 91-100 
};

__constant GGfloat PP3[101] = { 0.0f, 
  1.53728e-16f, 2.95909e-16f, 1.95042e-15f, 6.24521e-16f, 4.69459e-17f, 3.1394e-17f, 2.38808e-17f, 3.59428e-16f, 1.2947e-17f, 1.01182e-17f,     // 1-10 
  6.99543e-18f, 6.5138e-18f, 5.24063e-18f, 4.12831e-18f, 4.22067e-18f, 2.12802e-16f, 3.27035e-16f, 2.27705e-16f, 1.86943e-15f, 8.10577e-16f,     // 11-20 
  1.80541e-15f, 9.32266e-16f, 5.93459e-16f, 4.93049e-16f, 5.03211e-19f, 2.38223e-16f, 4.5181e-16f, 5.34468e-16f, 5.16504e-16f, 3.0641e-16f,     // 21-30 
  1.24646e-18f, 2.13805e-16f, 1.21448e-18f, 2.02122e-16f, 5.91556e-18f, 3.4609e-18f, 1.39331e-18f, 5.47242e-18f, 1.71017e-18f, 7.92438e-19f,     // 31-40 
  4.72225e-19f, 2.74825e-19f, 4.02137e-18f, 1.6662e-16f, 1.68841e-16f, 4.73202e-19f, 7.28319e-19f, 3.64382e-15f, 1.53323e-19f, 4.15409e-18f,     // 41-50 
  7.91645e-18f, 6.54036e-19f, 1.04123e-17f, 9.116e-17f, 5.97268e-19f, 1.23272e-15f, 5.83259e-18f, 5.42458e-18f, 2.20137e-17f, 1.19654e-17f,     // 51-60 
  2.3481e-17f, 1.53337e-17f, 8.38225e-16f, 3.40248e-17f, 3.50901e-17f, 1.95115e-17f, 2.91803e-16f, 1.98684e-17f, 3.59425e-16f, 1.54e-17f,     // 61-70 
  3.04174e-17f, 2.71295e-17f, 2.6803e-17f, 2.36469e-17f, 2.56818e-17f, 2.50364e-17f, 2.6818e-17f, 2.56229e-17f, 2.7419e-17f, 2.27442e-17f,     // 71-80 
  1.38078e-15f, 1.49595e-15f, 1.20023e-16f, 1.74446e-15f, 1.82836e-15f, 5.80108e-18f, 3.02324e-17f, 3.71029e-17f, 1.01058e-16f, 4.87707e-16f,     // 81-90 
  4.18953e-17f, 4.03182e-17f, 1.11553e-16f, 9.51125e-16f, 2.57569e-15f, 1.14294e-15f, 2.98597e-15f, 5.88714e-16f, 1.46196e-20f, 1.53226e-15f     // 91-100 
};

__constant GGfloat PP4[101] = { 0.0f, 
  1.10561e-15f, 3.50254e-16f, 1.56836e-16f, 7.86286e-15f, 2.2706e-16f, 7.28454e-16f, 4.54123e-16f, 8.03792e-17f, 4.91833e-16f, 1.45891e-16f,     // 1-10 
  1.71829e-16f, 3.90707e-15f, 2.76487e-15f, 4.345e-16f, 6.80131e-16f, 4.04186e-16f, 8.95703e-17f, 3.32136e-16f, 1.3847e-17f, 4.16869e-17f,     // 11-20 
  1.37963e-17f, 1.96187e-17f, 2.93852e-17f, 2.46581e-17f, 4.49944e-16f, 3.80311e-16f, 1.62925e-15f, 7.52449e-16f, 9.45445e-16f, 5.47652e-16f,     // 21-30 
  6.89379e-16f, 1.37078e-15f, 1.22209e-15f, 1.13856e-15f, 9.06914e-16f, 8.77868e-16f, 9.70871e-16f, 1.8532e-16f, 1.69254e-16f, 1.14059e-15f,     // 31-40 
  7.90712e-16f, 5.36611e-16f, 8.27932e-16f, 2.4329e-16f, 5.82899e-16f, 1.97595e-16f, 1.96263e-16f, 1.73961e-16f, 1.62174e-16f, 5.31143e-16f,     // 41-50 
  5.29731e-16f, 4.1976e-16f, 4.91842e-16f, 4.67937e-16f, 4.32264e-16f, 6.91046e-17f, 1.62962e-16f, 9.87241e-16f, 1.04526e-15f, 1.05819e-15f,     // 51-60 
  1.10579e-16f, 1.49116e-16f, 4.61021e-17f, 1.5143e-16f, 1.53667e-16f, 1.67844e-15f, 2.7494e-17f, 2.31253e-16f, 2.27211e-15f, 1.33401e-15f,     // 61-70 
  9.02548e-16f, 1.77743e-15f, 1.76608e-15f, 9.45054e-16f, 1.06805e-16f, 1.06085e-16f, 1.01688e-16f, 1.0226e-16f, 7.7793e-16f, 8.0166e-16f,     // 71-80 
  9.18595e-17f, 2.73428e-17f, 3.01222e-17f, 3.09814e-17f, 3.39028e-17f, 1.49653e-15f, 1.19511e-15f, 1.40408e-15f, 2.37226e-15f, 8.35973e-17f,     // 81-90 
  1.4089e-15f, 1.2819e-15f, 4.96925e-17f, 6.04886e-17f, 7.39507e-17f, 6.6832e-17f, 1.09433e-16f, 9.61804e-17f, 1.38525e-16f, 2.49104e-16f     // 91-100 
};

__constant GGfloat PP5[101] = { 0.0f, 
  6.89413e-17f, 2.11456e-17f, 2.47782e-17f, 7.01557e-17f, 1.01544e-15f, 1.76177e-16f, 1.28191e-16f, 1.80511e-17f, 1.96803e-16f, 3.16753e-16f,     // 1-10 
  1.21362e-15f, 6.6366e-17f, 8.42625e-17f, 1.01935e-16f, 1.34162e-16f, 1.87076e-18f, 2.76259e-17f, 1.2217e-16f, 1.66059e-18f, 1.76249e-18f,     // 11-20 
  1.13734e-18f, 1.58963e-18f, 1.33987e-18f, 1.18496e-18f, 2.44536e-16f, 6.69957e-19f, 2.5667e-17f, 2.62482e-17f, 2.55816e-17f, 2.6574e-17f,     // 21-30 
  2.26522e-16f, 2.17703e-18f, 2.07434e-16f, 8.8717e-19f, 1.75583e-16f, 1.81312e-16f, 1.83716e-16f, 2.58371e-15f, 1.74416e-15f, 1.7473e-16f,     // 31-40 
  1.76817e-16f, 1.74757e-16f, 1.6739e-16f, 2.68691e-19f, 1.8138e-19f, 1.60726e-16f, 1.59441e-16f, 1.36927e-16f, 2.70127e-16f, 1.63371e-16f,     // 41-50 
  1.29776e-16f, 1.49012e-16f, 1.17301e-16f, 1.67919e-17f, 1.47596e-16f, 1.14246e-19f, 1.10392e-15f, 1.58755e-16f, 1.11706e-16f, 1.80135e-16f,     // 51-60 
  1.00213e-15f, 9.44133e-16f, 4.722e-20f, 1.18997e-15f, 1.16311e-15f, 2.31716e-16f, 1.86238e-15f, 1.53632e-15f, 2.45853e-17f, 2.08069e-16f,     // 61-70 
  1.08659e-16f, 1.29019e-16f, 1.24987e-16f, 1.07865e-16f, 1.03501e-15f, 1.05211e-15f, 9.38473e-16f, 8.66912e-16f, 9.3778e-17f, 9.91467e-17f,     // 71-80 
  2.58481e-17f, 9.72329e-17f, 9.77921e-16f, 1.02928e-16f, 1.01767e-16f, 1.81276e-16f, 1.07026e-16f, 1.11273e-16f, 3.25695e-17f, 1.77629e-15f,     // 81-90 
  1.18382e-16f, 1.111e-16f, 1.56996e-15f, 8.45221e-17f, 3.6783e-16f, 1.20652e-16f, 3.91104e-16f, 3.52282e-15f, 4.29979e-16f, 1.28308e-16f     // 91-100 
};

__constant GGfloat PP6[101] = { 0.0f, 
  6.57834f, 3.91446f, 7.59547f, 10.707f, 3.97317f, 4.00593f, 3.93206f, 8.10644f, 3.97743f, 4.04641f,     // 1-10 
  4.30202f, 4.19399f, 4.27399f, 4.4169f, 4.04829f, 2.21745f, 11.3523f, 1.84976f, 1.61905f, 3.68297f,     // 11-20 
  1.5704f, 2.58852f, 3.59827f, 3.61633f, 9.07174f, 1.76738f, 1.97272f, 1.91032f, 1.9838f, 2.64286f,     // 21-30 
  4.16296f, 1.80149f, 3.94257f, 1.72731f, 2.27523f, 2.57383f, 3.33453f, 2.2361f, 2.94376f, 3.91332f,     // 31-40 
  5.01832f, 6.8016f, 2.19508f, 1.65926f, 1.63781f, 4.23097f, 3.4399f, 2.55583f, 7.96814f, 2.06573f,     // 41-50 
  1.84175f, 3.23516f, 1.79129f, 2.90259f, 3.18266f, 1.51305f, 1.88361f, 1.91925f, 1.68033f, 1.72078f,     // 51-60 
  1.66246f, 1.66676f, 1.49394f, 1.58924f, 1.57558f, 1.63307f, 1.84447f, 1.60296f, 1.56719f, 1.62166f,     // 61-70 
  1.5753f, 1.57329f, 1.558f, 1.57567f, 1.55612f, 1.54607f, 1.53251f, 1.51928f, 1.50265f, 1.52445f,     // 71-80 
  1.4929f, 1.51098f, 2.52959f, 1.42334f, 1.41292f, 2.0125f, 1.45015f, 1.43067f, 2.6026f, 1.39261f,     // 81-90 
  1.38559f, 1.37575f, 2.53155f, 2.51924f, 1.32386f, 2.31791f, 2.47722f, 1.33584f, 9.60979f, 6.84949f     // 91-100 
};

__constant GGfloat PP7[101] = { 0.0f, 
  3.99983f, 6.63093f, 3.85593f, 1.69342f, 14.7911f, 7.03995f, 8.89527f, 13.1929f, 4.93354f, 5.59461f,     // 1-10 
  3.98033f, 1.74578f, 2.67629f, 14.184f, 8.88775f, 13.1809f, 4.51627f, 13.7677f, 9.53727f, 4.04257f,     // 11-20 
  7.88725f, 5.78566f, 4.08148f, 4.18194f, 7.96292f, 8.38322f, 3.31429f, 13.106f, 13.0857f, 13.1053f,     // 21-30 
  3.54708f, 2.08567f, 2.38131f, 2.58162f, 3.199f, 3.20493f, 3.19799f, 1.88697f, 1.80323f, 3.15596f,     // 31-40 
  4.10675f, 5.68928f, 3.93024f, 11.2607f, 4.86595f, 12.1708f, 12.2867f, 9.29496f, 1.61249f, 5.0998f,     // 41-50 
  5.25068f, 6.67673f, 5.82498f, 6.12968f, 6.94532f, 1.71622f, 1.63028f, 3.34945f, 2.84671f, 2.66325f,     // 51-60 
  2.73395f, 1.93715f, 1.72497f, 2.74504f, 2.71531f, 1.52039f, 1.58191f, 1.61444f, 2.67701f, 1.51369f,     // 61-70 
  2.60766f, 1.46608f, 1.49792f, 2.49166f, 2.84906f, 2.80604f, 2.92788f, 2.76411f, 2.59305f, 2.5855f,     // 71-80 
  2.80503f, 1.4866f, 1.46649f, 1.45595f, 1.44374f, 1.54865f, 2.45661f, 2.43268f, 1.35352f, 1.35911f,     // 81-90 
  2.26339f, 2.26838f, 1.35877f, 1.37826f, 1.3499f, 1.36574f, 1.33654f, 1.33001f, 1.37648f, 4.28173f     // 91-100 
};

__constant GGfloat PP8[101] = { 0.0f, 
  4.0f, 4.0f, 5.94686f, 4.10265f, 7.87177f, 12.0509f, 12.0472f, 3.90597f, 5.34338f, 6.33072f,     // 1-10 
  2.76777f, 7.90099f, 5.58323f, 4.26372f, 3.3005f, 5.69179f, 2.3698f, 3.68167f, 5.2807f, 4.61212f,     // 11-20 
  5.87809f, 4.46207f, 4.59278f, 4.67584f, 1.75212f, 7.00575f, 2.05428f, 2.00415f, 2.02048f, 1.98413f,     // 21-30 
  1.71725f, 3.18743f, 1.74231f, 4.40997f, 2.01626f, 1.8622f, 1.7544f, 1.60332f, 2.23338f, 1.70932f,     // 31-40 
  1.67223f, 1.64655f, 1.76198f, 6.33416f, 7.92665f, 1.67835f, 1.67408f, 1.55895f, 9.3642f, 1.68776f,     // 41-50 
  2.02167f, 1.65401f, 2.20616f, 1.76498f, 1.63064f, 7.13771f, 3.17033f, 1.65236f, 2.66943f, 1.62703f,     // 51-60 
  2.72469f, 2.73686f, 10.86f, 2.76759f, 2.69728f, 1.62436f, 2.76662f, 1.48514f, 1.57342f, 1.61518f,     // 61-70 
  3.18455f, 2.73467f, 2.72521f, 2.786f, 2.35611f, 2.31574f, 2.5787f, 2.46877f, 2.89052f, 2.6478f,     // 71-80 
  1.50419f, 2.73998f, 2.79809f, 2.66207f, 2.73089f, 1.34835f, 2.59656f, 2.7006f, 1.41867f, 4.26255f,     // 81-90 
  2.47985f, 2.47126f, 1.72573f, 3.44856f, 1.36451f, 2.8715f, 2.35731f, 1.28196f, 4.1224f, 1.32633f     // 91-100 
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline void KleinNishinaComptonSampleSecondaries(__global GGEMSPrimaryParticles* primary_particle, __global GGEMSRandom* random, __global GGEMSMaterialTables const* materials, __global GGEMSParticleCrossSections const* particle_cross_sections, GGuchar const index_material, GGint const index_particle)
  \param primary_particle - buffer of particles
  \param random - pointer on random numbers
  \param materials - buffer of materials
  \param particle_cross_sections - pointer to cross sections activated in navigator
  \param index_material - index of the material
  \param index_particle - index of the particle
  \brief Klein Nishina Compton model, Effects due to binding of atomic electrons are negliged.
*/
inline void LivermoreRayleighSampleSecondaries(
  __global GGEMSPrimaryParticles* primary_particle,
  __global GGEMSRandom* random,
  __global GGEMSMaterialTables const* materials,
  __global GGEMSParticleCrossSections const* particle_cross_sections,
  GGuchar const index_material,
  GGint const index_particle
)
{
  GGfloat const kE0 = 0.009952493733686183f; //primary_particle->E_[index_particle];

  if (kE0 <= 250.0e-6f) { // 250 eV
    primary_particle->status_[index_particle] = DEAD;
    return;
  }

  // Current Direction
  GGfloat3 const kGammaDirection = {
    primary_particle->dx_[index_particle],
    primary_particle->dy_[index_particle],
    primary_particle->dz_[index_particle]
  };

  GGuint const kNumberOfBins = particle_cross_sections->number_of_bins_;
  GGuchar const kNEltsMinusOne = materials->number_of_chemical_elements_[index_material]-1;
  GGushort const kMixtureID = materials->index_of_chemical_elements_[index_material];
  GGuint const kEnergyID = primary_particle->E_index_[index_particle];

  // Get last atom
  GGuchar selected_atomic_number_z = materials->atomic_number_Z_[kMixtureID+kNEltsMinusOne];

  // Select randomly one element that composed the material
  GGuchar i = 0;
  if (kNEltsMinusOne > 0) {
    // Get Cross Section of Livermore Rayleigh
    GGfloat const kCS = LinearInterpolation(
      particle_cross_sections->energy_bins_[kEnergyID],
      particle_cross_sections->photon_cross_sections_[RAYLEIGH_SCATTERING][kEnergyID + kNumberOfBins*index_material],
      particle_cross_sections->energy_bins_[kEnergyID+1],
      particle_cross_sections->photon_cross_sections_[RAYLEIGH_SCATTERING][kEnergyID+1 + kNumberOfBins*index_material],
      kE0
    );

    // Get a random
    GGfloat const x = KissUniform(random, index_particle) * kCS;

    GGfloat cross_section = 0.0f;
    while (i < kNEltsMinusOne) {
      GGuchar atomic_number_z = materials->atomic_number_Z_[kMixtureID+i];
      cross_section += materials->atomic_number_density_[kMixtureID+i] * LinearInterpolation(
        particle_cross_sections->energy_bins_[kEnergyID],
        particle_cross_sections->photon_cross_sections_per_atom_[RAYLEIGH_SCATTERING][kEnergyID + kNumberOfBins*atomic_number_z],
        particle_cross_sections->energy_bins_[kEnergyID+1],
        particle_cross_sections->photon_cross_sections_per_atom_[RAYLEIGH_SCATTERING][kEnergyID+1 + kNumberOfBins*atomic_number_z],
        kE0
      );

      if (x < cross_section) {
        selected_atomic_number_z = atomic_number_z;
        break;
      }
      ++i;
    }
  }

  // Sample the angle of the scattered photon
  GGfloat const kXX = FACTOR*kE0*kE0;

  GGfloat const kN0 = PP6[selected_atomic_number_z] - 1.0f;
  GGfloat const kN1 = PP7[selected_atomic_number_z] - 1.0f;
  GGfloat const kN2 = PP8[selected_atomic_number_z] - 1.0f;
  GGfloat const kB0 = PP3[selected_atomic_number_z];
  GGfloat const kB1 = PP4[selected_atomic_number_z];
  GGfloat const kB2 = PP5[selected_atomic_number_z];

  GGfloat x = 2.0f*kXX*kB0;
  GGfloat const kW0 = (x < 0.02f) ? kN0*x*(1.0f - 0.5f*(kN0 - 1.0f)*x*(1.0f - (kN0 - 2.0f)*x/3.0f))
    : 1.0f - exp(-kN0*log(1.0f + x)); 

  x = 2.0f*kXX*kB1;
  GGfloat const kW1 = (x < 0.02f) ? kN1*x*(1.0f - 0.5f*(kN1 - 1.0f)*x*(1.0f - (kN1 - 2.0f)*x/3.0f))
    : 1.0f - exp(-kN1*log(1.0f + x));
 
  x = 2.0f*kXX*kB2;
  GGfloat const kW2 = (x < 0.02f) ? kN2*x*(1.0f - 0.5f*(kN2 - 1.0f)*x*(1.0f - (kN2 - 2.0f)*x/3.0f))
    : 1.0f - exp(-kN2*log(1.0f + x));

  GGfloat const kX0 = kW0*PP0[selected_atomic_number_z]/(kB0*kN0);
  GGfloat const kX1 = kW1*PP1[selected_atomic_number_z]/(kB1*kN1);
  GGfloat const kX2 = kW2*PP2[selected_atomic_number_z]/(kB2*kN2);

  GGfloat costheta = 0.0f;
  do {
    GGfloat w = kW0;
    GGfloat n = kN0;
    GGfloat b = kB0;

    x = KissUniform(random, index_particle)*(kX0+kX1+kX2);
    if (x > kX0) {
      x -= kX0;
      if (x <= kX1) {
        w = kW1;
        n = kN1;
        b = kB1;
      }
      else {
        w = kW2;
        n = kN2;
        b = kB2;
      }
    }
    n = 1.0f/n;

    // sampling of angle
    GGfloat y = KissUniform(random, index_particle)*w;
    if (y < 0.02f) {
      x = y*n*(1.0f + 0.5f*(n + 1.0f)*y*(1.0f - (n + 2.0f)*y/3.0f));
    }
    else {
      x = exp(-n*log(1.0f - y)) - 1.0f;
    }

    costheta = 1.0f - x/(b*kXX);
  } while (2.0f*KissUniform(random, index_particle) > 1.0f + costheta*costheta || costheta < -1.0f);

  GGfloat phi  = TWO_PI * KissUniform(random, index_particle);
  GGfloat sintheta = sqrt((1.0f - costheta)*(1.0f + costheta));

  GGfloat3 gamma_direction = {sintheta*cos(phi), sintheta*sin(phi), costheta};
  gamma_direction = RotateUnitZ(gamma_direction, kGammaDirection);
  gamma_direction = GGfloat3UnitVector(gamma_direction);

  // Update direction
  primary_particle->dx_[index_particle] = gamma_direction.x;
  primary_particle->dy_[index_particle] = gamma_direction.y;
  primary_particle->dz_[index_particle] = gamma_direction.z;

  #ifdef GGEMS_TRACKING
  if (index_particle == primary_particle->particle_tracking_id) {
    printf("\n");
    printf("[GGEMS OpenCL function LivermoreRayleighSampleSecondaries]     Photon energy: %e keV\n", kE0/keV);
    printf("[GGEMS OpenCL function LivermoreRayleighSampleSecondaries]     Photon direction: %e %e %e\n", kGammaDirection.x, kGammaDirection.y, kGammaDirection.z);
    printf("[GGEMS OpenCL function LivermoreRayleighSampleSecondaries]     Number of element in material %s: %d\n", particle_cross_sections->material_names_[index_material], materials->number_of_chemical_elements_[index_material]);
    printf("[GGEMS OpenCL function LivermoreRayleighSampleSecondaries]     Selected element: %u\n", selected_atomic_number_z);
    printf("[GGEMS OpenCL function LivermoreRayleighSampleSecondaries]     Scattered photon direction: %e %e %e\n", primary_particle->dx_[index_particle], primary_particle->dy_[index_particle], primary_particle->dz_[index_particle]);
  }
  #endif
}

#endif

#endif // GUARD_GGEMS_PHYSICS_GGEMSRAYLEIGHSCATTERINGMODELS_HH
