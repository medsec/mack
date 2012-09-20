#include <mack/options/parser.hpp>
#include <mack/options/values.hpp>
#include <mack/options/exit_requested.hpp>
#include <cstdlib>

#include <string>
#include <mack/core/cracker.hpp>
#include <mack/callback.hpp>
#include <mack/target_loader.hpp>

/**
 * @program{mack_the_knife}
 * @brief Cracking targets.
 * @details See \ref page_crackers_and_algorithms for an explanation of the concepts.
 * @type_option{C,cracker,crackers}
 * Sets the cracker to be used.
 * @type_option{o,output,callbacks,mack::callbacks::Console_Output}
 * Sets the callback for found targets.
 * @type_option{t,loader,targetloaders,mack::targetloaders::Default_Loader}
 * Sets the target loader to load your target file the right way.
 */
int
main(int argc, char** argv)
{
  try
  {
    mack::options::parser parser(argc, argv, "mack_the_knife");
    while (true)
    {
      mack::options::values const* program_values = parser.parse();
      mack::core::Cracker* cracker =
        program_values->get<mack::core::Cracker>("cracker");
      mack::callbacks::Callback* callback =
        program_values->get<mack::callbacks::Callback>("output");
      mack::targetloaders::Target_Loader* target_loader=
    	program_values->get<mack::targetloaders::Target_Loader>("loader");

      cracker->crack(callback, target_loader);

      delete cracker;
      delete callback;
      delete target_loader;
    }
  }
  catch (mack::options::exit_requested const& e)
  {
    return e.exit_code;
  }
}

