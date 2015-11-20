// GGEMS Copyright (C) 2015

/*!
 * \file txt_reader.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 19 novembre 2015
 *
 *
 *
 */

#ifndef TXT_READER_CU
#define TXT_READER_CU

#include "txt_reader.cuh"

/////:: Private

// Remove all white space
std::string TxtReader::m_remove_white_space(std::string txt) {
    txt.erase(remove_if(txt.begin(), txt.end(), isspace), txt.end());
    return txt;
}

// Read the list of tokens in a txt line
std::vector< std::string > TxtReader::m_split_txt(std::string line) {
    /*
    std::istringstream iss(line);
    std::vector< std::string > tokens{std::istream_iterator<std::string>{iss},
                                      std::istream_iterator<std::string>{}};
    return tokens;
    */

    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::copy(std::istream_iterator<std::string>(iss),
         std::istream_iterator<std::string>(),
         std::back_inserter(tokens));

    return tokens;

}

/////:: Main functions

// Skip comment starting with "#"
void TxtReader::skip_comment(std::istream & is) {
    i8 c;
    i8 line[1024];
    if (is.eof()) return;
    is >> c;
    while (is && (c=='#')) {
        is.getline(line, 1024);
        is >> c;
        if (is.eof()) return;
    }
    is.unget();
}

/// Range file

// Read start range
f32 TxtReader::read_start_range(std::string txt) {
    f32 res;
    txt = txt.substr(0, txt.find(" "));
    std::stringstream(txt) >> res;
    return res;
}

// Read stop range
f32 TxtReader::read_stop_range(std::string txt) {
    f32 res;
    txt = txt.substr(txt.find(" ")+1);
    txt = txt.substr(0, txt.find(" "));
    std::stringstream(txt) >> res;
    return res;
}

// Read material range
std::string TxtReader::read_mat_range(std::string txt) {
    txt = txt.substr(txt.find(" ")+1);
    txt = txt.substr(txt.find(" ")+1);
    return txt.substr(0, txt.find(" "));
}

/// MHD

// Read mhd key
std::string TxtReader::read_key(std::string txt) {
    txt = txt.substr(0, txt.find("="));
    return m_remove_white_space(txt);
}

// Read string mhd arg
std::string TxtReader::read_key_string_arg(std::string txt) {
    txt = txt.substr(txt.find("=")+1);
    return m_remove_white_space(txt);
}

// Read i32 mhd arg
i32 TxtReader::read_key_i32_arg(std::string txt) {
    i32 res;
    txt = txt.substr(txt.find("=")+1);
    txt = m_remove_white_space(txt);
    std::stringstream(txt) >> res;
    return res;
}

// Read int mhd arg
i32 TxtReader::read_key_i32_arg_atpos(std::string txt, i32 pos) {
    i32 res;
    txt = txt.substr(txt.find("=")+2);
    if (pos==0) {
        txt = txt.substr(0, txt.find(" "));
    }
    if (pos==1) {
        txt = txt.substr(txt.find(" ")+1);
        txt = txt.substr(0, txt.find(" "));
    }
    if (pos==2) {
        txt = txt.substr(txt.find(" ")+1);
        txt = txt.substr(txt.find(" ")+1);
    }
    std::stringstream(txt) >> res;
    return res;
}

// Read f32 mhd arg
f32 TxtReader::read_key_f32_arg_atpos(std::string txt, i32 pos) {
    f32 res;
    txt = txt.substr(txt.find("=")+2);
    if (pos==0) {
        txt = txt.substr(0, txt.find(" "));
    }
    if (pos==1) {
        txt = txt.substr(txt.find(" ")+1);
        txt = txt.substr(0, txt.find(" "));
    }
    if (pos==2) {
        txt = txt.substr(txt.find(" ")+1);
        txt = txt.substr(txt.find(" ")+1);
    }
    std::stringstream(txt) >> res;
    return res;
}

/// Elts and mats data file

// Read material name
std::string TxtReader::read_material_name(std::string txt) {
    return txt.substr(0, txt.find(":"));
}

// Read material density
f32 TxtReader::read_material_density(std::string txt) {
    f32 res;
    // density
    txt = txt.substr(txt.find("d=")+2);
    std::string txt1 = txt.substr(0, txt.find(" "));
    std::stringstream(txt1) >> res;
    // unit
    txt = txt.substr(txt.find(" ")+1);
    txt = txt.substr(0, txt.find(";"));
    txt = m_remove_white_space(txt);
    if (txt=="g/cm3")  return res *gram/cm3;
    if (txt=="mg/cm3") return res *mg/cm3;
        printf("read densité %f\n",res);
    return res;

}

// Read material number of elements
ui16 TxtReader::read_material_nb_elements(std::string txt) {
    ui16 res;
    txt = txt.substr(txt.find("n=")+2);
    txt = txt.substr(0, txt.find(";"));
    txt = m_remove_white_space(txt);
    std::stringstream(txt) >> res;
    return res;
}

// Read material element name
std::string TxtReader::read_material_element(std::string txt) {
    txt = txt.substr(txt.find("name=")+5);
    txt = txt.substr(0, txt.find(";"));
    txt = m_remove_white_space(txt);
    return txt;
}

// Read material element fraction TODO Add compound definition
f32 TxtReader::read_material_fraction(std::string txt) {
    f32 res;
    txt = txt.substr(txt.find("f=")+2);
    txt = m_remove_white_space(txt);
    std::stringstream(txt) >> res;
    return res;
}

// Read element name
std::string TxtReader::read_element_name(std::string txt) {
    return txt.substr(0, txt.find(":"));
}

// Read element Z
i32 TxtReader::read_element_Z(std::string txt) {
    i32 res;
    txt = txt.substr(txt.find("Z=")+2);
    txt = txt.substr(0, txt.find("."));
    std::stringstream(txt) >> res;
    return res;
}

// Read element A
f32 TxtReader::read_element_A(std::string txt) {
    f32 res;
    txt = txt.substr(txt.find("A=")+2);
    txt = txt.substr(0, txt.find("g/mole"));
    std::stringstream(txt) >> res;
    return res *gram/mole;
}

/// General

// Read int in txt file
i32 TxtReader::read_i32_atpos(std::string txt, i32 pos) {
    i32 res;
    std::vector<std::string> args = m_split_txt(txt);

    if (pos < args.size()) {
        std::stringstream(args[pos]) >> res;
        return res;
    } else {
        print_warning("TxtReader: arg position out of range!");
        exit_simulation();
    }
}

// Read f32 in txt file
f32 TxtReader::read_f32_atpos(std::string txt, i32 pos) {
    f32 res;
    std::vector<std::string> args = m_split_txt(txt);

    if (pos < args.size()) {
        std::stringstream(args[pos]) >> res;
        return res;
    } else {
        print_warning("TxtReader: arg position out of range!");
        exit_simulation();
    }
}




#endif












