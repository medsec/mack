#ifndef __MACK_CORE_XML_PARSER_HPP__
#define __MACK_CORE_XML_PARSER_HPP__

#include <string>
#include <boost/filesystem.hpp>
#include <map>

namespace mack {
namespace core {

/**
 * @brief Type for the attributes of a xml tag.
 */
typedef std::map<std::string, std::string> attributes_type;

/**
 * @brief Base class for event based (<b>not</b> thread safe) parsers of xml
 * files.
 * @details This is a higher-level abstraction on top of the *expat* xml parsing
 * library. The parser is event driven. Thus, the xml document is parsed while
 * reading and the appropriate (protected) methods are invoked if a xml element
 * starts or ends, or characters are read
 * (see
 * \ref start_element(std::string const& name, attributes_type const& attributes),
 * \ref end_element(std::string const& name) and
 * \ref characters(std::string const& text)).
 *
 * Generally, a class extending this base class overwrites at least one of these
 * methods and provides further methods for retrieving the parsed information.
 *
 * @author Johannes Kiesel
 * @date Sep 15 2012
 */
class xml_parser
{

  public:

    /**
     * @brief The constructor.
     */
    xml_parser();

    /**
     * @brief The destructor.
     */
    virtual
    ~xml_parser();

    /**
     * @brief Parses given text as content of a xml file.
     * @details If this method is invoked, the parser will be <tt>clear</tt>ed
     * and the content will be parsed.
     * @param content the xml tree to be parsed
     * @throws mack::core::files::parse_error if an error occurred during
     * parsing
     * @see clear()
     */
    void
    parse(
        std::string const& content);

    /**
     * @brief Parses the file denoted by given file path.
     * @details If this method is invoked, the file will be read,
     * the parser will be <tt>clear</tt>ed
     * and the content will be parsed.
     * @param file_path the file path of the xml file to be parsed
     * @throws file_not_exists_error if no file with given file_path exists
     * @throws not_a_file_error if the object at given file_path is not a
     * regular file
     * @throws file_read_error if an error occured on reading the file
     * @throws mack::core::files::parse_error if an error occurred during
     * parsing
     * @see clear()
     */
    void
    parse_file(
        char const* file_path);

    /**
     * @brief Parses the file denoted by given file path.
     * @details If this method is invoked, the file will be read,
     * the parser will be <tt>clear</tt>ed
     * and the content will be parsed.
     * @param file_path the file path of the xml file to be parsed
     * @throws file_not_exists_error if no file with given file_path exists
     * @throws not_a_file_error if the object at given file_path is not a
     * regular file
     * @throws file_read_error if an error occured on reading the file
     * @throws mack::core::files::parse_error if an error occurred during
     * parsing
     * @see clear()
     */
    void
    parse_file(
        std::string const& file_path);

    /**
     * @brief Parses the file denoted by given file path.
     * @details If this method is invoked, the file will be read,
     * the parser will be <tt>clear</tt>ed
     * and the content will be parsed.
     * @param file_path the file path of the xml file to be parsed
     * @throws file_not_exists_error if no file with given file_path exists
     * @throws not_a_file_error if the object at given file_path is not a
     * regular file
     * @throws file_read_error if an error occured on reading the file
     * @throws mack::core::files::parse_error if an error occurred during
     * parsing
     * @see clear()
     */
    void
    parse_file(
        boost::filesystem::path const& file_path);

    /**
     * @brief Resets the internal state of this parser.
     */
    virtual
    void
    clear() = 0;

  protected:

    /**
     * @brief Method which is invoked if an xml start tag is encountered while
     * parsing the xml tree.
     * @details Note that if the xml element does not have any children
     * (<tt>\<<i>name</i> <b>/</b>\></tt>), the
     * \ref end_element(std::string const& name) method will be invoked
     * directly after this method is invoked.
     *
     * If not overridden, this method does nothing.
     * @param name the name of the encountered element
     * @param attributes the attributes of the encountered element
     */
    virtual
    void
    start_element(
        std::string const& name,
        attributes_type const& attributes);

    /**
     * @brief Method which is invoked if a xml element is closed.
     * @details Note that if the xml element does not have any children
     * (<tt>\<<i>name</i> <b>/</b>\></tt>), this method is invoked directly
     * after
     * \ref start_element(std::string const& name, attributes_type const& attributes).
     *
     * If not overridden, this method does nothing.
     * @param name the name of the encountered element
     */
    virtual
    void
    end_element(
        std::string const& name);

    /**
     * @brief Method which is invoked if characters are encountered while
     * parsing a xml tree.
     * @details Leading and trailing spaces are ommitted. Always, the complete
     * text between two xml tags is provided.
     *
     * If not overridden, this method does nothing.
     * @param text the encountered characters up to the next tag
     */
    virtual
    void
    characters(
        std::string const& text);

  private:

    void
    clear_character_buffer();

    void
    buffer_characters(
        const char* chars,
        const int len);

    static
    void
    static_start_element(
        void* userData,
        const char* name,
        const char** atts);

    static
    void
    static_end_element(
        void* userData,
        const char* name);

    static
    void
    static_characters(
        void* userData,
        const char* chars,
        const int len);

    std::string _character_buffer;

}; // class xml_parser

} // namespace core
} // namespace mack

#endif /* __MACK_CORE_XML_PARSER_HPP__ */

