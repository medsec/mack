#include <unittest++/UnitTest++.h>

#include <mack/options/program_options.hpp>
#include <mack/options/exceptions.hpp>
#include <mack/options/type_option.hpp>
#include <mack/core/null_pointer_error.hpp>
#include <mack/core/files.hpp>

#include <boost/exception/diagnostic_information.hpp>

#include <mack/options/types/test_option_type.hpp>

#include <string>

using namespace mack::options;

SUITE(mack_options_program_options)
{
  TEST(TestConstructorEmpty)
  {
    const std::vector<option*> opts_vector;

    const program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);
    CHECK(opts.is_empty());
  }

  TEST(TestConstructor)
  {
    option* opt1 = new option(
        "s",
        "long",
        "Brief.",
        "Detailed.");
    option* opt2 = new option(
        "s2",
        "long2",
        "Brief2.",
        "Detailed2.");

    std::vector<option*> opts_vector;
    opts_vector.push_back(opt1);
    opts_vector.push_back(opt2);

    const program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);
    CHECK(!opts.is_empty());
  }

  TEST(TestConstructorFailNullPointer)
  {
    option* opt1 = new option(
        "s",
        "long",
        "Brief.",
        "Detailed.");

    std::vector<option*> opts_vector;
    opts_vector.push_back(opt1);
    opts_vector.push_back(NULL);

    CHECK_THROW(const program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector),
        mack::core::null_pointer_error);
  }

  TEST(TestConstructorFailExternFlagCollision)
  {
    // short - long
    option* opt1 = new option(
        "s",
        "long",
        "Brief.",
        "Detailed.");
    option* opt2 = new option(
        "s2",
        "s",
        "Brief2.",
        "Detailed2.");

    std::vector<option*> opts_vector;
    opts_vector.push_back(opt1);
    opts_vector.push_back(opt2);

    CHECK_THROW(program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector),
        flag_collision_error);

    // short - short
    opt1 = new option(
        "s",
        "long",
        "Brief.",
        "Detailed.");
    opt2 = new option(
        "s",
        "long2",
        "Brief2.",
        "Detailed2.");

    opts_vector.clear();
    opts_vector.push_back(opt1);
    opts_vector.push_back(opt2);

    CHECK_THROW(program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector),
        flag_collision_error);

    // long - short
    opt1 = new option(
        "s",
        "long",
        "Brief.",
        "Detailed.");
    opt2 = new option(
        "long",
        "long2",
        "Brief2.",
        "Detailed2.");

    opts_vector.clear();
    opts_vector.push_back(opt1);
    opts_vector.push_back(opt2);

    CHECK_THROW(program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector),
        flag_collision_error);

    // long - long
    opt1 = new option(
        "s",
        "long",
        "Brief.",
        "Detailed.");
    opt2 = new option(
        "s2",
        "long",
        "Brief2.",
        "Detailed2.");

    opts_vector.clear();
    opts_vector.push_back(opt1);
    opts_vector.push_back(opt2);

    CHECK_THROW(program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector),
        flag_collision_error);
  }

  TEST(TestConstructorFailDefaultOptionFlagCollision)
  {
    // short - long
    option* opt1 = new option(
        "help",
        "long",
        "Brief.",
        "Detailed.");
    option* opt2 = new option(
        "s2",
        "long2",
        "Brief2.",
        "Detailed2.");

    std::vector<option*> opts_vector;
    opts_vector.push_back(opt1);
    opts_vector.push_back(opt2);

    CHECK_THROW(program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector),
        flag_collision_error);

    // short - short
    opt1 = new option(
        "s",
        "long",
        "Brief.",
        "Detailed.");
    opt2 = new option(
        "h",
        "long2",
        "Brief2.",
        "Detailed2.");

    opts_vector.clear();
    opts_vector.push_back(opt1);
    opts_vector.push_back(opt2);

    CHECK_THROW(program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector),
        flag_collision_error);

    // long - short
    opt1 = new option(
        "s",
        "dL",
        "Brief.",
        "Detailed.");
    opt2 = new option(
        "s2",
        "long2",
        "Brief2.",
        "Detailed2.");

    opts_vector.clear();
    opts_vector.push_back(opt1);
    opts_vector.push_back(opt2);

    CHECK_THROW(program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector),
        flag_collision_error);

    // long - long
    opt1 = new option(
        "s",
        "long",
        "Brief.",
        "Detailed.");
    opt2 = new option(
        "s2",
        "verbosity",
        "Brief2.",
        "Detailed2.");

    opts_vector.clear();
    opts_vector.push_back(opt1);
    opts_vector.push_back(opt2);

    CHECK_THROW(program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector),
        flag_collision_error);
  }

  TEST(TestConstructorFailInternFlagCollision)
  {
    option* opt1 = new option(
        "s",
        "long",
        "Brief.",
        "Detailed.");
    option* opt2 = new option(
        "s2",
        "s2",
        "Brief2.",
        "Detailed2.");

    std::vector<option*> opts_vector;
    opts_vector.push_back(opt1);
    opts_vector.push_back(opt2);

    CHECK_THROW(const program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector),
        flag_collision_error);
  }

  TEST(TestGetProgramName)
  {
    std::vector<option*> opts_vector;
    const program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);
    CHECK_EQUAL("my_program", opts.get_program_name());
  }

  TEST(TestGetBriefDescription)
  {
    std::vector<option*> opts_vector;
    const program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);
    CHECK_EQUAL("s and s2.", opts.get_brief_description());
  }

  TEST(TestGetFullDescription)
  {
    std::vector<option*> opts_vector;
    const program_options opts(
        "my_program",
        "s and s2.",
        "long and long2.",
        opts_vector);
    CHECK_EQUAL("s and s2.\nlong and long2.", opts.get_full_description());
  }

  TEST(TestGetFullDescriptionNoDetailed)
  {
    std::vector<option*> opts_vector;
    const program_options opts(
        "my_program",
        "s and s2.",
        "",
        opts_vector);
    CHECK_EQUAL("s and s2.", opts.get_full_description());
  }

  TEST(TestIteration)
  {
    option* opt1 = new option(
        "s",
        "long",
        "Brief.",
        "Detailed.");
    option* opt2 = new option(
        "s2",
        "long2",
        "Brief2.",
        "Detailed2.");

    std::vector<option*> opts_vector;
    opts_vector.push_back(opt1);
    opts_vector.push_back(opt2);

    program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    std::vector<option*>::iterator it = opts.begin();
    CHECK(!(it == opts.end()));
    CHECK(*it == opt1);

    ++it;
    CHECK(!(it == opts.end()));
    CHECK(*it == opt2);

    ++it;
    CHECK(it == opts.end());
  }

  TEST(TestConstIteration)
  {
    option* opt1 = new option(
        "s",
        "long",
        "Brief.",
        "Detailed.");
    option* opt2 = new option(
        "s2",
        "long2",
        "Brief2.",
        "Detailed2.");

    std::vector<option*> opts_vector;
    opts_vector.push_back(opt1);
    opts_vector.push_back(opt2);

    const program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    std::vector<option const*>::const_iterator it = opts.begin();
    CHECK(!(it == opts.end()));
    CHECK(*it == opt1);

    ++it;
    CHECK(!(it == opts.end()));
    CHECK(*it == opt2);

    ++it;
    CHECK(it == opts.end());
  }

  TEST(TestEmptyIteration)
  {
    std::vector<option*> opts_vector;
    program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    CHECK(opts.begin() == opts.end());
  }

  TEST(TestEmptyConstIteration)
  {
    std::vector<option*> opts_vector;
    const program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    CHECK(opts.begin() == opts.end());
  }

  TEST(TestGetOption)
  {
    option* opt1 = new option(
        "s",
        "long",
        "Brief.",
        "Detailed.");
    option* opt2 = new option(
        "s2",
        "long2",
        "Brief2.",
        "Detailed2.");

    std::vector<option*> opts_vector;
    opts_vector.push_back(opt1);
    opts_vector.push_back(opt2);

    program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    CHECK(opts.get_option(std::string("s")) == opt1);
    CHECK(opts.get_option("long") == opt1);
    CHECK(opts.get_option("s2") == opt2);
    CHECK(opts.get_option("long2") == opt2);
  }

  TEST(TestGetOptionFail)
  {
    option* opt1 = new option(
        "s",
        "long",
        "Brief.",
        "Detailed.");
    option* opt2 = new option(
        "s2",
        "long2",
        "Brief2.",
        "Detailed2.");

    std::vector<option*> opts_vector;
    opts_vector.push_back(opt1);
    opts_vector.push_back(opt2);

    program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    CHECK_THROW(opts.get_option("none"), no_such_option_error);
  }

  TEST(TestGetConstOption)
  {
    option* opt1 = new option(
        "s",
        "long",
        "Brief.",
        "Detailed.");
    option* opt2 = new option(
        "s2",
        "long2",
        "Brief2.",
        "Detailed2.");

    std::vector<option*> opts_vector;
    opts_vector.push_back(opt1);
    opts_vector.push_back(opt2);

    const program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    CHECK(opts.get_option("s") == opt1);
    CHECK(opts.get_option("long") == opt1);
    CHECK(opts.get_option("s2") == opt2);
    CHECK(opts.get_option("long2") == opt2);
  }

  TEST(TestGetConstOptionFail)
  {
    option* opt1 = new option(
        "s",
        "long",
        "Brief.",
        "Detailed.");
    option* opt2 = new option(
        "s2",
        "long2",
        "Brief2.",
        "Detailed2.");

    std::vector<option*> opts_vector;
    opts_vector.push_back(opt1);
    opts_vector.push_back(opt2);

    const program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    CHECK_THROW(opts.get_option("none"), no_such_option_error);
  }

  TEST(TestCreate)
  {
    option* opt1 = new option(
        "s1",
        "long1",
        "Brief.",
        "Detailed.");
    option* opt2 = new option(
        "s2",
        "long2",
        "Brief.",
        "Detailed.",
        "default");
    option* opt3 = new option(
        "s3",
        "long3",
        "Brief.",
        "Detailed.",
        "default",
        "initial");

    std::vector<option*> opts_vector;
    opts_vector.push_back(opt1);
    opts_vector.push_back(opt2);
    opts_vector.push_back(opt3);

    program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    opts.get_option("s1")->set_value("value1");
    opts.get_option("s2")->set_value("value2");
    opts.get_option("s3")->set_value("value3");

    values const* vals = opts.create();
    CHECK_EQUAL("value1", vals->get("long1"));
    CHECK_EQUAL("value2", vals->get("long2"));
    CHECK_EQUAL("value3", vals->get("long3"));
    delete vals;

    opts.get_option("s2")->set_value();
    opts.get_option("s3")->set_value();

    vals = opts.create();
    CHECK_EQUAL("value1", vals->get("long1"));
    CHECK_EQUAL("default", vals->get("long2"));
    CHECK_EQUAL("default", vals->get("long3"));
    delete vals;
  }

  TEST(TestCreateFailNoValue)
  {
    option* opt1 = new option(
        "s1",
        "long1",
        "Brief.",
        "Detailed.");
    option* opt2 = new option(
        "s2",
        "long2",
        "Brief.",
        "Detailed.",
        "default");
    option* opt3 = new option(
        "s3",
        "long3",
        "Brief.",
        "Detailed.",
        "default",
        "initial");

    std::vector<option*> opts_vector;
    opts_vector.push_back(opt2);
    opts_vector.push_back(opt1);
    opts_vector.push_back(opt3);

    program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    CHECK_THROW(opts.create(), no_value_error);
  }

  TEST(TestSet)
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

    program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    opts.set("help");

    opts.set("s1", "recursive_class");
    opts.set("s1.s", "empty_class");
    opts.set("s1.default", "empty_class");
    opts.set("long1.i", "empty_class");

    opts.set("s2", "tests::mack::options::ambiguous_class");
    opts.set("long2.standard", "option 1");
    opts.set("long2.d", "option 2");
    opts.set("s2.i");
    opts.set("b");

    CHECK(opts.is_help_set());
    CHECK_EQUAL(
        "tests::mack::options::recursive_class",
        opts.get_option("long1")->get_value());
    CHECK_EQUAL(
        "tests::mack::options::empty_class",
        opts.get_option("long1")->get_child_option("standard")->get_value());
    CHECK_EQUAL(
        "tests::mack::options::empty_class",
        opts.get_option("long1")->get_child_option("default")->get_value());
    CHECK_EQUAL(
        "tests::mack::options::empty_class",
        opts.get_option("long1")->get_child_option("initial")->get_value());
    CHECK_EQUAL(
        "tests::mack::options::ambiguous_class",
        opts.get_option("long2")->get_value());
    CHECK_EQUAL(
        "option 1",
        opts.get_option("long2")->get_child_option("standard")->get_value());
    CHECK_EQUAL(
        "option 2",
        opts.get_option("long2")->get_child_option("default")->get_value());
    CHECK_EQUAL(
        "option 2",
        opts.get_option("long2")->get_child_option("initial")->get_value());
    CHECK_EQUAL(
        "true",
        opts.get_option("long2")->get_child_option("boolean")->get_value());
  }

  TEST(TestSetFailInvalidFlag)
  {
    option* opt1 = new type_option(
        "s1",
        "long1",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type);

    std::vector<option*> opts_vector;
    opts_vector.push_back(opt1);

    program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    CHECK_THROW(opts.set("s1._default", "value"), invalid_flag_error);
    CHECK_THROW(opts.set("_default", "value"), invalid_flag_error);
  }

  TEST(TestSetFailNoSuchNamespace)
  {
    option* opt1 = new type_option(
        "s1",
        "long1",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type);

    std::vector<option*> opts_vector;
    opts_vector.push_back(opt1);

    program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    CHECK_THROW(opts.set("s2.default", "value"), no_such_namespace_error);
  }

  TEST(TestSetFailNoSuchNamespaceNoChildren)
  {
    option* opt1 = new type_option(
        "s1",
        "long1",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type);

    std::vector<option*> opts_vector;
    opts_vector.push_back(opt1);

    program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    CHECK_THROW(opts.set("s1.default.opt", "value"), no_such_namespace_error);
  }

  TEST(TestSetFailNoSuchOption)
  {
    option* opt1 = new type_option(
        "s1",
        "long1",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type,
        "tests::mack::options::ambiguous_class");

    std::vector<option*> opts_vector;
    opts_vector.push_back(opt1);

    program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    CHECK_THROW(opts.set("s1.default_", "value"), no_such_option_error);
  }

  TEST(TestSetFailNoSuchOptionNoChildren)
  {
    option* opt1 = new type_option(
        "s1",
        "long1",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type);

    std::vector<option*> opts_vector;
    opts_vector.push_back(opt1);

    program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    CHECK_THROW(opts.set("s1.default_", "value"), no_such_option_error);
    CHECK_THROW(opts.set("long2", "value"), no_such_option_error);
  }

  TEST(TestSetFailAmbiguousOption)
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

    program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    opts.set("s1", "recursive_class");

    opts.set("s2", "tests::mack::options::ambiguous_class");

    CHECK_THROW(opts.set("default", "empty_class"), option_collision_error);
  }

  TEST(TestSetFailNotAmbiguousOption)
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
    option* opt3 = new option(
        "s3",
        "default",
        "Brief.",
        "Detailed.");

    std::vector<option*> opts_vector;
    opts_vector.push_back(opt1);
    opts_vector.push_back(opt2);
    opts_vector.push_back(opt3);

    program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    opts.set("s1", "recursive_class");

    opts.set("s2", "tests::mack::options::ambiguous_class");

    opts.set("default", "empty_class");

    CHECK_EQUAL("empty_class", opts.get_option("default")->get_value());
  }

  TEST(TestSetFromConfiguration)
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

    program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    opts.set_from_configuration(std::string()
        + "help\n"
        + " s1 =  recursive_class  \n"
        + "s1.s = empty_class # my comment\n"
        + "s1.default = empty_class\n"
        + " # Comment\n"
        + "long1.i = empty_class\n"
        + "  \n"
        + "s2 = tests::mack::options::ambiguous_class\n"
        + "long2.standard = option 1\n"
        + "long2.d = option 2 \n"
        + "s2.i\n"
        + "b\n"
        );

    CHECK(opts.is_help_set());
    CHECK_EQUAL(
        "tests::mack::options::recursive_class",
        opts.get_option("long1")->get_value());
    CHECK_EQUAL(
        "tests::mack::options::empty_class",
        opts.get_option("long1")->get_child_option("standard")->get_value());
    CHECK_EQUAL(
        "tests::mack::options::empty_class",
        opts.get_option("long1")->get_child_option("default")->get_value());
    CHECK_EQUAL(
        "tests::mack::options::empty_class",
        opts.get_option("long1")->get_child_option("initial")->get_value());
    CHECK_EQUAL(
        "tests::mack::options::ambiguous_class",
        opts.get_option("long2")->get_value());
    CHECK_EQUAL(
        "option 1",
        opts.get_option("long2")->get_child_option("standard")->get_value());
    CHECK_EQUAL(
        "option 2",
        opts.get_option("long2")->get_child_option("default")->get_value());
    CHECK_EQUAL(
        "option 2",
        opts.get_option("long2")->get_child_option("initial")->get_value());
    CHECK_EQUAL(
        "true",
        opts.get_option("long2")->get_child_option("boolean")->get_value());
  }

  TEST(TestSetFromConfigurationFile)
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

    program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    opts.set_from_configuration_file("tests/resources/configuration_file.txt");

    CHECK(opts.is_help_set());
    CHECK_EQUAL(
        "tests::mack::options::recursive_class",
        opts.get_option("long1")->get_value());
    CHECK_EQUAL(
        "tests::mack::options::empty_class",
        opts.get_option("long1")->get_child_option("standard")->get_value());
    CHECK_EQUAL(
        "tests::mack::options::empty_class",
        opts.get_option("long1")->get_child_option("default")->get_value());
    CHECK_EQUAL(
        "tests::mack::options::empty_class",
        opts.get_option("long1")->get_child_option("initial")->get_value());
    CHECK_EQUAL(
        "tests::mack::options::ambiguous_class",
        opts.get_option("long2")->get_value());
    CHECK_EQUAL(
        "option 1",
        opts.get_option("long2")->get_child_option("standard")->get_value());
    CHECK_EQUAL(
        "option 2",
        opts.get_option("long2")->get_child_option("default")->get_value());
    CHECK_EQUAL(
        "option 2",
        opts.get_option("long2")->get_child_option("initial")->get_value());
    CHECK_EQUAL(
        "true",
        opts.get_option("long2")->get_child_option("boolean")->get_value());
  }

  TEST(TestSetFromConfigurationFileByOption)
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

    program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    opts.set("c", "tests/resources/configuration_file.txt");

    CHECK(opts.is_help_set());
    CHECK_EQUAL(
        "tests::mack::options::recursive_class",
        opts.get_option("long1")->get_value());
    CHECK_EQUAL(
        "tests::mack::options::empty_class",
        opts.get_option("long1")->get_child_option("standard")->get_value());
    CHECK_EQUAL(
        "tests::mack::options::empty_class",
        opts.get_option("long1")->get_child_option("default")->get_value());
    CHECK_EQUAL(
        "tests::mack::options::empty_class",
        opts.get_option("long1")->get_child_option("initial")->get_value());
    CHECK_EQUAL(
        "tests::mack::options::ambiguous_class",
        opts.get_option("long2")->get_value());
    CHECK_EQUAL(
        "option 1",
        opts.get_option("long2")->get_child_option("standard")->get_value());
    CHECK_EQUAL(
        "option 2",
        opts.get_option("long2")->get_child_option("default")->get_value());
    CHECK_EQUAL(
        "option 2",
        opts.get_option("long2")->get_child_option("initial")->get_value());
    CHECK_EQUAL(
        "true",
        opts.get_option("long2")->get_child_option("boolean")->get_value());
  }

  TEST(TestSetFromConfigurationFileByOptionFailFileNotExists)
  {
    std::vector<option*> opts_vector;
    program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    CHECK_THROW(
        opts.set("c", "tests/resources/does_not_exists.txt"),
        mack::core::files::file_not_exists_error);
  }

  TEST(TestSetFromConfigurationFileByOptionFailNotAFile)
  {
    std::vector<option*> opts_vector;
    program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    CHECK_THROW(
        opts.set("c", "tests/resources/directory"),
        mack::core::files::not_a_file_error);
  }

  TEST(TestSetFromConfigurationFileByOptionFailNoValue)
  {
    std::vector<option*> opts_vector;
    program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    CHECK_THROW(
        opts.set("c"),
        no_value_error);
  }

  TEST(TestSetAll)
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

    program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    char const* argv[15];
    argv[0] = "my_program";
    argv[1] = "--help";
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

    opts.set_all(15, argv, 1);

    CHECK(opts.is_help_set());
    CHECK_EQUAL(
        "tests::mack::options::recursive_class",
        opts.get_option("long1")->get_value());
    CHECK_EQUAL(
        "tests::mack::options::empty_class",
        opts.get_option("long1")->get_child_option("standard")->get_value());
    CHECK_EQUAL(
        "tests::mack::options::empty_class",
        opts.get_option("long1")->get_child_option("default")->get_value());
    CHECK_EQUAL(
        "tests::mack::options::empty_class",
        opts.get_option("long1")->get_child_option("initial")->get_value());
    CHECK_EQUAL(
        "tests::mack::options::ambiguous_class",
        opts.get_option("long2")->get_value());
    CHECK_EQUAL(
        "option 1",
        opts.get_option("long2")->get_child_option("standard")->get_value());
    CHECK_EQUAL(
        "option 2",
        opts.get_option("long2")->get_child_option("default")->get_value());
    CHECK_EQUAL(
        "option 2",
        opts.get_option("long2")->get_child_option("initial")->get_value());
    CHECK_EQUAL(
        "true",
        opts.get_option("long2")->get_child_option("boolean")->get_value());
  }

  TEST(TestSetAllConfigurationFile)
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

    program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    char const* argv[4];
    argv[0] = "my_program";
    argv[1] = "-c";
    argv[2] = "tests/resources/configuration_file.txt";
    argv[3] = "--help=false";

    opts.set_all(4, argv, 1);

    CHECK(!opts.is_help_set());
    CHECK_EQUAL(
        "tests::mack::options::recursive_class",
        opts.get_option("long1")->get_value());
    CHECK_EQUAL(
        "tests::mack::options::empty_class",
        opts.get_option("long1")->get_child_option("standard")->get_value());
    CHECK_EQUAL(
        "tests::mack::options::empty_class",
        opts.get_option("long1")->get_child_option("default")->get_value());
    CHECK_EQUAL(
        "tests::mack::options::empty_class",
        opts.get_option("long1")->get_child_option("initial")->get_value());
    CHECK_EQUAL(
        "tests::mack::options::ambiguous_class",
        opts.get_option("long2")->get_value());
    CHECK_EQUAL(
        "option 1",
        opts.get_option("long2")->get_child_option("standard")->get_value());
    CHECK_EQUAL(
        "option 2",
        opts.get_option("long2")->get_child_option("default")->get_value());
    CHECK_EQUAL(
        "option 2",
        opts.get_option("long2")->get_child_option("initial")->get_value());
    CHECK_EQUAL(
        "true",
        opts.get_option("long2")->get_child_option("boolean")->get_value());
  }


  TEST(TestSetAllFailUnboundValue)
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

    program_options opts(
        "my_program",
        "s and s2.",
        "long and long2",
        opts_vector);

    char const* argv[4];
    argv[0] = "my_program";
    argv[1] = "c";
    argv[2] = "tests/resources/configuration_file.txt";
    argv[3] = "--help=false";

    CHECK_THROW(opts.set_all(4, argv, 1), unbound_value_error);
  }
}

