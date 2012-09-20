#include <unittest++/UnitTest++.h>

#include <mack/options/option_switch.hpp>
#include <mack/options/exceptions.hpp>

#include <string>

using namespace mack::options;

SUITE(mack_options_option_switch)
{
  TEST(TestConstructor)
  {
    option_switch opt(
        "s",
        "long",
        "Brief.",
        "Detailed.");

    CHECK(opt.has_value());
    CHECK_EQUAL("false", opt.get_value());
  }

  TEST(TestSetValue)
  {
    option_switch opt(
        "s",
        "long",
        "Brief.",
        "Detailed.");

    opt.set_value();
    CHECK(opt.has_value());
    CHECK_EQUAL("true", opt.get_value());

    opt.set_value("false");
    CHECK(opt.has_value());
    CHECK_EQUAL("false", opt.get_value());
  }

  TEST(TestSetValueFailInvalidValue)
  {
    option_switch opt(
        "s",
        "long",
        "Brief.",
        "Detailed.");

    CHECK_THROW(opt.set_value("other"), invalid_value_error);
  }

  TEST(TestIsSelection)
  {
    option_switch opt(
        "s",
        "long",
        "Brief.",
        "Detailed.");

    CHECK(opt.is_selection());
  }

  TEST(TestSelectionIterator)
  {
    option_switch opt(
        "s",
        "long",
        "Brief.",
        "Detailed.");

    std::vector<std::string>::const_iterator it;

    it = opt.selection_values_begin();
    CHECK(it != opt.selection_values_end());
    CHECK_EQUAL(*it, "true");
    it++;
    CHECK(it != opt.selection_values_end());
    CHECK_EQUAL(*it, "false");
    it++;
    CHECK(it == opt.selection_values_end());
  }

  TEST(TestGetShortFlag)
  {
    option_switch opt(
        "s",
        "long",
        "Brief.",
        "Detailed.");

    CHECK_EQUAL("s", opt.get_short_flag());
  }

  TEST(TestGetLongFlag)
  {
    option_switch opt(
        "s",
        "long",
        "Brief.",
        "Detailed.");

    CHECK_EQUAL("long", opt.get_long_flag());
  }

  TEST(TestGetBrief)
  {
    option_switch opt(
        "s",
        "long",
        "Brief.",
        "Detailed.");

    CHECK_EQUAL("Brief.", opt.get_brief_description());
  }

  TEST(TestGetFullNoDetailed)
  {
    option_switch opt(
        "s",
        "long",
        "Brief.",
        "");

    CHECK_EQUAL("Brief.", opt.get_full_description());
  }

  TEST(TestGetFull)
  {
    option_switch opt(
        "s",
        "long",
        "Brief.",
        "Detailed.");

    CHECK_EQUAL("Brief.\nDetailed.", opt.get_full_description());
  }

  TEST(TestCreate)
  {
    option_switch opt(
        "s",
        "long",
        "Brief.",
        "Detailed.");

    value const* val = opt.create();
    CHECK(!val->get_boolean());

    opt.set_value();
    value const* val2 = opt.create();
    CHECK(val2->get_boolean());

    opt.set_value("true");
    value const* val3 = opt.create();
    CHECK(val3->get_boolean());

    opt.set_value();
    value const* val4 = opt.create();
    CHECK(val4->get_boolean());

    opt.set_value("false");
    value const* val5 = opt.create();
    CHECK(!val5->get_boolean());

    delete val;
    delete val2;
    delete val3;
    delete val4;
    delete val5;
  }

  TEST(TestHasChildOptions)
  {
    option_switch opt(
        "s",
        "long",
        "Brief.",
        "Detailed.");

    CHECK(!opt.has_child_options());
  }

  TEST(TestFailGetChildOptions)
  {
    option_switch opt(
        "s",
        "long",
        "Brief.",
        "Detailed.");

    CHECK_THROW(opt.get_child_options(), no_such_namespace_error);
  }

  TEST(TestFailGetChildOption)
  {
    option_switch opt(
        "s",
        "long",
        "Brief.",
        "Detailed.");

    CHECK_THROW(opt.get_child_option("flag"), no_such_namespace_error);
  }

  TEST(TestIsValidValue)
  {
    option_switch opt(
        "s",
        "long",
        "Brief.",
        "Detailed.");

    CHECK(opt.is_valid_value("true"));
    CHECK(opt.is_valid_value("false"));

    CHECK(!opt.is_valid_value(""));

    CHECK(!opt.is_valid_value("none"));
  }
}

