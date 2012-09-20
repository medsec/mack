#include <unittest++/UnitTest++.h>

#include <mack/options/options.hpp>
#include <mack/options/exceptions.hpp>
#include <mack/core/null_pointer_error.hpp>

#include <string>

using namespace mack::options;

SUITE(mack_options_options)
{
  TEST(TestConstructorEmpty)
  {
    const std::vector<option*> opts_vector;

    const options opts(
        "s and s2.",
        "long and long2",
        opts_vector);
    CHECK(opts.is_empty());
  }

  TEST(TestConstructor)
  {
    option* opt1 = new option(
        "h",
        "long",
        "Brief.",
        "Detailed.");
    option* opt2 = new option(
        "s2",
        "help",
        "Brief2.",
        "Detailed2.");

    std::vector<option*> opts_vector;
    opts_vector.push_back(opt1);
    opts_vector.push_back(opt2);

    const options opts(
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

    CHECK_THROW(const options opts(
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

    CHECK_THROW(options opts(
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

    CHECK_THROW(options opts(
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

    CHECK_THROW(options opts(
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

    CHECK_THROW(options opts(
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

    CHECK_THROW(const options opts(
        "s and s2.",
        "long and long2",
        opts_vector),
        flag_collision_error);
  }

  TEST(TestGetBriefDescription)
  {
    std::vector<option*> opts_vector;
    const options opts(
        "s and s2.",
        "long and long2",
        opts_vector);
    CHECK_EQUAL("s and s2.", opts.get_brief_description());
  }

  TEST(TestGetFullDescription)
  {
    std::vector<option*> opts_vector;
    const options opts(
        "s and s2.",
        "long and long2.",
        opts_vector);
    CHECK_EQUAL("s and s2.\nlong and long2.", opts.get_full_description());
  }

  TEST(TestGetFullDescriptionNoDetailed)
  {
    std::vector<option*> opts_vector;
    const options opts(
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

    options opts(
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

    const options opts(
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
    options opts(
        "s and s2.",
        "long and long2",
        opts_vector);

    CHECK(opts.begin() == opts.end());
  }

  TEST(TestEmptyConstIteration)
  {
    std::vector<option*> opts_vector;
    const options opts(
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

    options opts(
        "s and s2.",
        "long and long2",
        opts_vector);

    CHECK(opts.get_option("s") == opt1);
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

    options opts(
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

    const options opts(
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

    const options opts(
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

    options opts(
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

    options opts(
        "s and s2.",
        "long and long2",
        opts_vector);

    CHECK_THROW(opts.create(), no_value_error);
  }
}

