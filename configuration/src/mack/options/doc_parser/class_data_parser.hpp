#ifndef __MACK_OPTIONS_DOC_PARSER_CLASS_DATA_PARSER__
#define __MACK_OPTIONS_DOC_PARSER_CLASS_DATA_PARSER__

#include <mack/core/xml_parser.hpp>
#include <mack/options/doc_parser/data_structs.hpp>

#include <exception>
#include <boost/exception/exception.hpp>
#include <boost/exception/error_info.hpp>
#include <boost/exception/info.hpp>

#include <boost/filesystem.hpp>
#include <vector>

namespace mack {
namespace options {
namespace doc_parser {

class class_data_parser : public mack::core::xml_parser
{
  public:

    class_data_parser(
        boost::filesystem::path const& src_directory,
        boost::filesystem::path const& xml_directory);

    virtual
    ~class_data_parser();

    virtual
    void
    clear();

    void
    set_target_class_id(
        std::string const& target_class_id);

    class_data const&
    get_class_data() const;

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

    std::string _src_dir;

    boost::filesystem::path _xml_dir;

    std::string _target_class_id;

    class_data _data;

    option_data _option;

    std::vector<std::string> _stack;

    bool _get_name;

    bool _get_extends;

    bool _get_tparam;

    bool _get_brief;

    bool _is_simple_sect;

    bool _get_detailed;

    bool _get_option_text;

}; // class class_data_parser

} // namespace doc_parser
} // namespace options
} // namespace mack

#endif /* __MACK_OPTIONS_DOC_PARSER_CLASS_DATA_PARSER__ */

