#ifndef __MACK_OPTIONS_DOC_PARSER_TYPE_DATA_PARSER__
#define __MACK_OPTIONS_DOC_PARSER_TYPE_DATA_PARSER__

#include <mack/core/xml_parser.hpp>
#include <mack/options/doc_parser/data_structs.hpp>
#include <map>

namespace mack {
namespace options {
namespace doc_parser {

struct not_a_class_error : virtual std::exception, virtual boost::exception
{
};

typedef boost::error_info<struct tag_class_name, std::string>
  errinfo_class_name;

class type_data_parser : public mack::core::xml_parser
{
  public:

    type_data_parser(
        boost::filesystem::path const& src_directory,
        boost::filesystem::path const& xml_directory);

    virtual
    ~type_data_parser();

    virtual
    void
    clear();

    type_data const&
    get_type_data() const;

    std::vector<type_data> const&
    get_child_type_datas() const;

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

    boost::filesystem::path _src_dir;

    boost::filesystem::path _xml_dir;

    type_data _data;

    std::vector<type_data> _children;

    bool _get_name;

    std::string _current_refid;

    std::map<std::string, std::string> _refids;

}; // class type_data_parser

} // namespace doc_parser
} // namespace options
} // namespace mack

#endif /* __MACK_OPTIONS_DOC_PARSER_TYPE_DATA_PARSER__ */

