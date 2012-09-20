#include <mack/options/doc_parser/file_handler.hpp>
#include <mack/options/doc_parser/type_data_parser.hpp>
#include <boost/exception/diagnostic_information.hpp>
#include <boost/exception/get_error_info.hpp>
#include <cstdlib>
#include <iostream>
#include <mack/core/files.hpp>

int
main(int argc, char** argv)
{
  try
  {
    mack::options::doc_parser::file_handler handler(
        boost::filesystem::path("src"),
        boost::filesystem::path("doc/xml"));
    handler.run();
    return EXIT_SUCCESS;
  }
  catch (mack::options::doc_parser::not_a_class_error const& e)
  {
    std::string const* class_name =
      boost::get_error_info<mack::options::doc_parser::errinfo_class_name>(e);
    std::string const* type_name =
      boost::get_error_info<mack::options::doc_parser::errinfo_type_name>(e);
    std::cerr << "An option type class was defined ";
    if (type_name != NULL)
    {
      std::cerr << "for the type \"" << (*type_name) << "\" ";
    }
    std::cerr << "which is not known to doxygen";
    if (class_name != NULL)
    {
      std::cerr << ": \"" << (*class_name) << "\"";
    }
    std::cerr << std::endl;
  }
  catch (mack::core::files::file_not_exists_error const& e)
  {
    std::cerr << "File does not exist: "
      << *(boost::get_error_info<mack::core::files::errinfo_file>(e))
      << std::endl;
  }
  return EXIT_FAILURE;
}

