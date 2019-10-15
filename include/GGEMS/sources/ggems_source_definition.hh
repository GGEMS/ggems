#ifndef GUARD_GGEMS_SOURCES_GGEMSSOURCEDEFINITION_HH
#define GUARD_GGEMS_SOURCES_GGEMSSOURCEDEFINITION_HH

/*!
  \file ggems_source_definition.hh

  \brief GGEMS class managing the source in GGEMS, every new sources in GGEMS
  inherit from this class

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday October 15, 2019
*/

/*!
  \class GGEMSSourceDefinition
  \brief GGEMS class managing the source in GGEMS, every new sources in GGEMS
  inherit from this class
*/
class GGEMSSourceDefinition
{
  public:
    /*!
      \brief GGEMSSourceDefinition constructor
    */
    GGEMSSourceDefinition(void);

    /*!
      \brief GGEMSSourceDefinition destructor
    */
    ~GGEMSSourceDefinition(void);

  public:
    /*!
      \fn GGEMSSourceDefinition(GGEMSSourceDefinition const& ggems_source) = delete
      \param ggems_manager - reference on the ggems manager
      \brief Avoid copy of the class by reference
    */
    GGEMSSourceDefinition(GGEMSSourceDefinition const& ggems_source) = delete;

    /*!
      \fn GGEMSSourceDefinition& operator=(GGEMSSourceDefinition const& ggems_source) = delete
      \param ggems_manager - reference on the ggems manager
      \brief Avoid assignement of the class by reference
    */
    GGEMSSourceDefinition& operator=(
      GGEMSSourceDefinition const& ggems_source) = delete;

    /*!
      \fn GGEMSSourceDefinition(GGEMSSourceDefinition const&& ggems_source) = delete
      \param ggems_manager - rvalue reference on the ggems manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSSourceDefinition(GGEMSSourceDefinition const&& ggems_source) = delete;

    /*!
      \fn GGEMSSourceDefinition& operator=(GGEMSSourceDefinition const&& ggems_source) = delete
      \param ggems_manager - rvalue reference on the ggems manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSSourceDefinition& operator=(
      GGEMSSourceDefinition const&& ggems_source) = delete;
};

#endif // End of GUARD_GGEMS_SOURCES_GGEMSSOURCEDEFINITION_HH
