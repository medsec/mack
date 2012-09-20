#ifndef __MACK_OPTIONS_DOC_PARSER_PROGRAM_DATA_PARSER__
#define __MACK_OPTIONS_DOC_PARSER_PROGRAM_DATA_PARSER__

#include <mack/core/xml_parser.hpp>
#include <mack/options/doc_parser/data_structs.hpp>

#include <boost/filesystem.hpp>
#include <vector>

namespace mack {
namespace options {
namespace doc_parser {

class program_data_parser : public mack::core::xml_parser
{
  public:

    program_data_parser();

    virtual
    ~program_data_parser();

    virtual
    void
    clear();

    program_data const&
    get_program_data() const;

  protected:

    virtual
    void
    start_element(
        std::string const& name,
        mack::core::attributes_type const& attributes);

    virtual
    void
    end_element(
        std::string const& name);

    virtual
    void
    characters(
        std::string const& text);

  private:

    program_data _data;

    option_data _option;

    bool _get_name;

    bool _get_brief;

    bool _get_details;

    bool _is_simple_sect;

    bool _get_option_text;

}; // class program_data_parser

} // namespace doc_parser
} // namespace options
} // namespace mack

#endif /* __MACK_OPTIONS_DOC_PARSER_PROGRAM_DATA_PARSER__ */

