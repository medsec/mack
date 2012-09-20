#include <mack/options/parser.hpp>
#include <mack/options/values.hpp>
#include <mack/options/exit_requested.hpp>
#include <cstdlib>

/**
 * @program{my_program}
 * @brief A command line program.
 * @details This program will be automatically compiled
 * into the *bin* folder. The executable has the same name
 * as this file without extension.
 *
 * @type_option{C,class,my_class_type}
 * This option is mandatory and requires a class of type
 * *my_class_type*.
 */
int
main(int argc, char** argv)
{
  try
  {
    mack::options::parser parser(argc, argv, "my_program");
    while (true) // multiple settings can be run
    {
      mack::options::values const* program_values = parser.parse();
      // main program starts here (if exit was not requested)...
      my_class_type_class* instance =
        program_values->get<my_class_type_class>("class");
      ...
      delete instance;
      // ... and ends here
    }
  }
  catch (mack::options::exit_requested e)
  {
    return e.exit_code;
  }
}
