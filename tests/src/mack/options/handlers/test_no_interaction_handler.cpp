#include <unittest++/UnitTest++.h>

#include <mack/options/handlers/no_interaction_handler.hpp>
#include <mack/options/exceptions.hpp>
#include <mack/core/null_pointer_error.hpp>

#include <mack/options/exit_requested.hpp>
#include <cstdlib>

#include <string>
#include <sstream>

using namespace mack::options;

SUITE(mack_options_handlers_no_interaction_handler)
{
  TEST(TestConstructing)
  {
    const std::vector<option*> opts_vector;
    program_options* opts = new program_options(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    handlers::no_interaction_handler handler;
    handler.set_program_options(opts);
  }

  TEST(TestConstructingFailNullPointer)
  {
    handlers::no_interaction_handler handler;
    CHECK_THROW(
        handler.set_program_options(NULL),
        mack::core::null_pointer_error);
  }

  TEST(TestRun)
  {
    option* opt = new option(
        "s",
        "long",
        "Brief.",
        "Detailed.");

    std::vector<option*> opts_vector;
    opts_vector.push_back(opt);

    program_options* opts = new program_options(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    opts->set("long", "value");

    std::stringstream ss;
    handlers::no_interaction_handler handler(ss);
    handler.set_program_options(opts);

    values const* vals = handler.run();
    CHECK_EQUAL("value", vals->get("long"));

    delete vals;
    delete opts;
  }

  TEST(TestRunFailRunTwice)
  {
    option* opt = new option(
        "s",
        "long",
        "Brief.",
        "Detailed.");

    std::vector<option*> opts_vector;
    opts_vector.push_back(opt);

    program_options* opts = new program_options(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    std::stringstream ss;
    handlers::no_interaction_handler handler(ss);
    handler.set_program_options(opts);

    opts->set("long", "value");

    values const* vals = handler.run();

    delete vals;

    try
    {
      delete handler.run();
      CHECK(false);
    }
    catch (exit_requested e)
    {
      CHECK(e.exit_code == EXIT_SUCCESS);
    }

    delete opts;
  }

  TEST(TestRunFailNullPointer)
  {
    handlers::no_interaction_handler handler;
    CHECK_THROW(
        handler.run(),
        mack::core::null_pointer_error);
  }

  TEST(TestRunFailNotSet)
  {
    option* opt = new option(
        "s",
        "long",
        "Brief.",
        "Detailed.");

    std::vector<option*> opts_vector;
    opts_vector.push_back(opt);

    program_options* opts = new program_options(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    std::stringstream ss;
    handlers::no_interaction_handler handler(ss);
    handler.set_program_options(opts);

    try
    {
      delete handler.run();
      CHECK(false);
    }
    catch (exit_requested e)
    {
      CHECK(e.exit_code == EXIT_FAILURE);
    }

    delete opts;
  }

}

