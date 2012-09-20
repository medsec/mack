#include "xml_parser.hpp"

#include <expat.h>

#include <boost/algorithm/string/trim.hpp>

#include <mack/core/files.hpp>

mack::core::xml_parser* CURRENT_PARSER = NULL;

mack::core::xml_parser::xml_parser()
  : _character_buffer()
{
}

mack::core::xml_parser::~xml_parser()
{
}

void
mack::core::xml_parser::parse(
    std::string const& content)
{
  // Clear old
  clear();
  _character_buffer.clear();

  // Initialization
  xml_parser* old_parser = CURRENT_PARSER;
  CURRENT_PARSER = this;
  XML_Parser parser = XML_ParserCreate(NULL);
  XML_SetStartElementHandler(parser, static_start_element);
  XML_SetEndElementHandler(parser, static_end_element);
  XML_SetCharacterDataHandler(parser, static_characters);

  // Parsing
  const int parse_exit = XML_Parse(parser, content.c_str(), content.length(), 1);

  // Cleaning up
  if (parser != NULL)
  {
    XML_ParserFree(parser);
  }
  CURRENT_PARSER = old_parser;

  if (!parse_exit)
  {
    // Failure on parsing
    BOOST_THROW_EXCEPTION(mack::core::files::parse_error());
  }
  else
  {
    // Parsing okay -> clear trailing characters
    clear_character_buffer();
  }
}

void
mack::core::xml_parser::parse_file(
    char const* file_path)
{
  parse_file(std::string(file_path));
}

void
mack::core::xml_parser::parse_file(
    std::string const& file_path)
{
  parse_file(boost::filesystem::path(file_path));
}

void
mack::core::xml_parser::parse_file(
    boost::filesystem::path const& file_path)
{
  clear();
  try
  {
    parse(mack::core::files::read_file(file_path));
  }
  catch (mack::core::files::parse_error e)
  {
    e << mack::core::files::errinfo_file(file_path.native());
    throw;
  }
}

void
mack::core::xml_parser::start_element(
    std::string const& name,
    attributes_type const& attributes)
{
  // Do nothing
}

void
mack::core::xml_parser::end_element(
    std::string const& name)
{
  // Do nothing
}

void
mack::core::xml_parser::characters(
    std::string const& text)
{
  // Do nothing
}

void
mack::core::xml_parser::clear_character_buffer()
{
  if (!boost::algorithm::trim_copy(_character_buffer).empty())
  {
    characters(_character_buffer);
  }
  _character_buffer.clear();
}

void
mack::core::xml_parser::buffer_characters(
    const char* chars,
    const int len)
{
  _character_buffer.append(chars, len);
}

void XMLCALL
mack::core::xml_parser::static_start_element(
    void* userData,
    const char* name,
    const char** atts)
{
  CURRENT_PARSER->clear_character_buffer();
  attributes_type attributes;
  for (int i = 0; atts[i] != 0; i += 2)
  {
    attributes.insert(std::pair<std::string, std::string>(atts[i], atts[i + 1]));
  }
  CURRENT_PARSER->start_element(std::string(name), attributes);
}

void XMLCALL
mack::core::xml_parser::static_end_element(
    void* userData,
    const char* name)
{
  CURRENT_PARSER->clear_character_buffer();
  CURRENT_PARSER->end_element(std::string(name));
}

void XMLCALL
mack::core::xml_parser::static_characters(
    void* userData,
    const char* chars,
    const int len)
{
  CURRENT_PARSER->buffer_characters(chars, len);
}

