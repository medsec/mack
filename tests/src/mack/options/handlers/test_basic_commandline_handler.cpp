#include <unittest++/UnitTest++.h>

#include <mack/options/handlers/basic_commandline_handler.hpp>
#include <mack/options/exceptions.hpp>
#include <mack/core/null_pointer_error.hpp>

#include <mack/options/exit_requested.hpp>
#include <cstdlib>

#include <string>
#include <sstream>

using namespace mack::options;

SUITE(mack_options_handlers_basic_commandline_handler)
{
  TEST(TestConstructing)
  {
    const std::vector<option*> opts_vector;
    program_options* opts = new program_options(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    handlers::basic_commandline_handler handler;
    handler.set_program_options(opts);
  }

  TEST(TestConstructingFailNullPointer)
  {
    handlers::basic_commandline_handler handler;
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

    std::stringstream os;
    std::stringstream is;
    is << ":run" << std::endl;
    handlers::basic_commandline_handler handler(false, is, os, os);
    handler.set_program_options(opts);

    values const* vals = handler.run();
    CHECK_EQUAL("value", vals->get("long"));

    delete vals;
    delete opts;
  }

  TEST(TestRunRunTwice)
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

    std::stringstream os;
    std::stringstream is;
    is << ":run" << std::endl;
    is << ":run" << std::endl;
    handlers::basic_commandline_handler handler(true, is, os, os);
    handler.set_program_options(opts);

    opts->set("long", "value");

    values const* vals = handler.run();
    CHECK_EQUAL("value", vals->get("long"));
    delete vals;

    vals = handler.run();
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

    std::stringstream os;
    std::stringstream is;
    is << ":run" << std::endl;
    is << ":run" << std::endl;
    handlers::basic_commandline_handler handler(false, is, os, os);
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
    std::stringstream os;
    std::stringstream is;
    is << ":run" << std::endl;
    handlers::basic_commandline_handler handler(false, is, os, os);
    CHECK_THROW(
        handler.run(),
        mack::core::null_pointer_error);
  }

  TEST(TestRunAndSet)
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

    std::stringstream os;
    std::stringstream is;
    is << ":run" << std::endl; // Not yet set
    is << "long = value" << std::endl; // set
    is << ":run" << std::endl; // run correctly
    handlers::basic_commandline_handler handler(false, is, os, os);
    handler.set_program_options(opts);

    values const* vals = handler.run();
    CHECK_EQUAL("value", vals->get("long"));

    delete vals;
    delete opts;
  }

  TEST(TestRunQuit)
  {
    std::stringstream os;
    std::stringstream is;
    is << ":quit" << std::endl;
    handlers::basic_commandline_handler handler(false, is, os, os);

    try
    {
      delete handler.run();
      CHECK(false);
    }
    catch (exit_requested e)
    {
      CHECK(e.exit_code == EXIT_SUCCESS);
    }
  }
}

