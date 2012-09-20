#ifndef __MACK_OPTIONS_DOC_PARSER_LOCATION_PARSER__
#define __MACK_OPTIONS_DOC_PARSER_LOCATION_PARSER__

#include <mack/core/xml_parser.hpp>

#include <vector>

namespace mack {
namespace options {
namespace doc_parser {

class location_parser : public mack::core::xml_parser
{
  public:

    location_parser();

    virtual
    ~location_parser();

    void
    set_target_class_id(
        std::string const& target_class_id);

    virtual
    void
    clear();

    std::string const&
    get_location() const;

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

    std::string _target_class_id;

    std::vector<std::string> _stack;

    std::string _location;

}; // class location_parser

} // namespace doc_parser
} // namespace options
} // namespace mack

#endif /* __MACK_OPTIONS_DOC_PARSER_LOCATION_PARSER__ */

