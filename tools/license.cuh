// GGEMS Copyright (C) 2015

/*!
 * \file license.h
 * \brief Header of the license source class
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 4 janvier 2016
 *
 *
 *
 */

#ifndef LICENSE_CUH
#define LICENSE_CUH

#include "global.cuh"

typedef struct
{
    ui32 erk[64];     /* encryption round keys */
    ui32 drk[64];     /* decryption round keys */
    int nr;           /* number of rounds */
}
aes_context;

struct LicenseData {
    std::string institution;

    std::string start_day;
    std::string start_month;
    std::string start_year;

    std::string expired_day;
    std::string expired_month;
    std::string expired_year;

    bool img_feature;
    bool dose_feature;

    bool clearence;
    bool read;
};

class License
{
public:
    License();
    ~License();

    /* void write_license ( std::string outputname, std::string institution,
                         std::string start_day, std::string start_month, std::string start_year,
                         std::string end_day, std::string end_month, std::string end_year ); */
    void read_license ( std::string inputname );
    void check_license();

    // License Data
    LicenseData info;

private:

    void m_print_word ( std::string txt, ui8 *aword, ui32 nbytes );

    // AES
    aes_context m_aes_ctx;
    ui8 m_aes_buf[16];

    /* decryption key schedule tables */
    int KT_init = 1;
    ui32 KT0[256];
    ui32 KT1[256];
    ui32 KT2[256];
    ui32 KT3[256];

    /* Functions */
    void m_aes_init_key();
    int m_aes_set_key ( aes_context *ctx, ui8 *key, ui32 nbits );
    void m_aes_encrypt ( aes_context *ctx, ui8 input[16], ui8 output[16] );
    void m_aes_decrypt ( aes_context *ctx, ui8 input[16], ui8 output[16] );

};

#endif
