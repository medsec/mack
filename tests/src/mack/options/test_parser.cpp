#include <unittest++/UnitTest++.h>

#include <mack/options/parser.hpp>
#include <mack/options/program_options.hpp>
#include <mack/options/type_option.hpp>
#include <mack/options/types/test_option_type.hpp>
#include <mack/options/handlers/basic_commandline_handler.hpp>
#include <mack/options/exceptions.hpp>
#include <mack/core/null_pointer_error.hpp>

#include <mack/options/exit_requested.hpp>
#include <cstdlib>

#include <string>
#include <sstream>

using namespace mack::options;

SUITE(mack_options_parser)
{
  TEST(TestInitialize)
  {
    option* opt1 = new type_option(
        "s1",
        "long1",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type);
    option* opt2 = new type_option(
        "s2",
        "long2",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type,
        "empty_class");

    std::vector<option*> opts_vector;
    opts_vector.push_back(opt1);
    opts_vector.push_back(opt2);

    program_options* opts = new program_options(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    char const* argv[15];
    argv[0] = "my_program";
    argv[1] = "--s1";
    argv[2] = "--s1";
    argv[3] = "recursive_class";
    argv[4] = "-s1.s=empty_class";
    argv[5] = "--s1.default";
    argv[6] = "empty_class";
    argv[7] = "-long1.i";
    argv[8] = "empty_class";
    argv[9] = "-s2=tests::mack::options::ambiguous_class";
    argv[10] = "--long2.standard=option 1";
    argv[11] = "--long2.d";
    argv[12] = "option 2";
    argv[13] = "-s2.i";
    argv[14] = "--b";

    std::stringstream message_out;
    std::stringstream error_out;
    parser parser(15, argv, opts, message_out, error_out);

    CHECK(!opts->is_help_set());
    CHECK_EQUAL(
        "tests::mack::options::recursive_class",
        opts->get_option("long1")->get_value());
    CHECK_EQUAL(
        "tests::mack::options::empty_class",
        opts->get_option("long1")->get_child_option("standard")->get_value());
    CHECK_EQUAL(
        "tests::mack::options::empty_class",
        opts->get_option("long1")->get_child_option("default")->get_value());
    CHECK_EQUAL(
        "tests::mack::options::empty_class",
        opts->get_option("long1")->get_child_option("initial")->get_value());
    CHECK_EQUAL(
        "tests::mack::options::ambiguous_class",
        opts->get_option("long2")->get_value());
    CHECK_EQUAL(
        "option 1",
        opts->get_option("long2")->get_child_option("standard")->get_value());
    CHECK_EQUAL(
        "option 2",
        opts->get_option("long2")->get_child_option("default")->get_value());
    CHECK_EQUAL(
        "option 2",
        opts->get_option("long2")->get_child_option("initial")->get_value());
    CHECK_EQUAL(
        "true",
        opts->get_option("long2")->get_child_option("boolean")->get_value());
  }

  TEST(TestConstructorFailNullPointer)
  {
    char const* argv[2];
    argv[0] = "my_program";
    argv[1] = "-b";

    std::stringstream message_out;
    std::stringstream error_out;
    CHECK_THROW(
        parser parser(2, argv, NULL, message_out, error_out),
        mack::core::null_pointer_error);
  }

  TEST(TestConstructorFailNoConfigurationFile)
  {
    std::vector<option*> opts_vector;

    program_options* opts = new program_options(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    char const* argv[2];
    argv[0] = "my_program";
    argv[1] = "-c";

    std::stringstream message_out;
    std::stringstream error_out;
    try
    {
      parser parser(2, argv, opts, message_out, error_out);
      CHECK(false);
    }
    catch (exit_requested e)
    {
      CHECK_EQUAL(EXIT_FAILURE, e.exit_code);
    }
  }

  TEST(TestConstructorHelp)
  {
    option* opt = new type_option(
        "s1",
        "long1",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type);
    std::vector<option*> opts_vector;
    opts_vector.push_back(opt);

    program_options* opts = new program_options(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    char const* argv[2];
    argv[0] = "my_program";
    argv[1] = "--help";

    std::stringstream message_out;
    std::stringstream error_out;
    try
    {
      parser parser(2, argv, opts, message_out, error_out);
      CHECK(false);
    }
    catch (exit_requested e)
    {
      CHECK_EQUAL(EXIT_SUCCESS, e.exit_code);
    }
  }

  TEST(TestCreate)
  {
    option* opt = new option(
        "s1",
        "long1",
        "Brief.",
        "Detailed.");
    std::vector<option*> opts_vector;
    opts_vector.push_back(opt);

    program_options* opts = new program_options(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    char const* argv[2];
    argv[0] = "my_program";
    argv[1] = "-s1=value";

    std::stringstream message_out;
    std::stringstream error_out;

    parser parser(2, argv, opts, message_out, error_out);

    values const* vals = parser.parse();

    CHECK_EQUAL("value", vals->get("long1"));
  }
}

