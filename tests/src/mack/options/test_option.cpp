#include <unittest++/UnitTest++.h>

#include <mack/options/option.hpp>
#include <mack/options/exceptions.hpp>

#include <string>

using namespace mack::options;

SUITE(mack_options_option)
{
  TEST(TestSimpleConstructor)
  {
    const option opt(
        "s",
        "long",
        "Brief.",
        "Detailed.");
    CHECK(!opt.has_value());
  }

  TEST(TestDefaultValueConstructor)
  {
    const option opt(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        "default");
    CHECK(opt.has_value());
    CHECK_EQUAL("default", opt.get_value());
  }

  TEST(TestDefaultStartValueConstructor)
  {
    const option opt(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        "default",
        "initial");
    CHECK(opt.has_value());
    CHECK_EQUAL("initial", opt.get_value());
  }

  TEST(TestGetShortFlag)
  {
    const option opt(
        "s",
        "long",
        "Brief.",
        "Detailed.");
    const option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        "default");
    const option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        "default",
        "initial");
    CHECK_EQUAL("s", opt.get_short_flag());
    CHECK_EQUAL("s", opt2.get_short_flag());
    CHECK_EQUAL("s", opt3.get_short_flag());
  }

  TEST(TestGetLongFlag)
  {
    const option opt(
        "s",
        "long",
        "Brief.",
        "Detailed.");
    const option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        "default");
    const option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        "default",
        "initial");
    CHECK_EQUAL("long", opt.get_long_flag());
    CHECK_EQUAL("long", opt2.get_long_flag());
    CHECK_EQUAL("long", opt3.get_long_flag());
  }

  TEST(TestGetBrief)
  {
    const option opt(
        "s",
        "long",
        "Brief.",
        "Detailed.");
    const option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        "default");
    const option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        "default",
        "initial");
    CHECK_EQUAL("Brief.", opt.get_brief_description());
    CHECK_EQUAL("Brief.", opt2.get_brief_description());
    CHECK_EQUAL("Brief.", opt3.get_brief_description());
  }

  TEST(TestGetFullNoDetailed)
  {
    const option opt(
        "s",
        "long",
        "Brief.",
        "");
    const option opt2(
        "s",
        "long",
        "Brief.",
        "",
        "default");
    const option opt3(
        "s",
        "long",
        "Brief.",
        "",
        "default",
        "initial");
    CHECK_EQUAL("Brief.", opt.get_full_description());
    CHECK_EQUAL("Brief.", opt2.get_full_description());
    CHECK_EQUAL("Brief.", opt3.get_full_description());
  }

  TEST(TestGetFull)
  {
    const option opt(
        "s",
        "long",
        "Brief.",
        "Detailed.");
    const option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        "default");
    const option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        "default",
        "initial");
    CHECK_EQUAL("Brief.\nDetailed.", opt.get_full_description());
    CHECK_EQUAL("Brief.\nDetailed.", opt2.get_full_description());
    CHECK_EQUAL("Brief.\nDetailed.", opt3.get_full_description());
  }

  TEST(TestSetGetValueEmpty)
  {
    option opt(
        "s",
        "long",
        "Brief.",
        "Detailed.");
    option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        "default");
    option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        "default",
        "initial");

    CHECK(!opt.has_value());
    CHECK(opt2.has_value());
    CHECK(opt3.has_value());

    CHECK_THROW(opt.get_value(), no_value_error);
    CHECK_EQUAL("default", opt2.get_value());
    CHECK_EQUAL("initial", opt3.get_value());

    opt.set_value();
    opt2.set_value();
    opt3.set_value();

    CHECK_THROW(opt.get_value(), no_value_error);
    CHECK_EQUAL("default", opt2.get_value());
    CHECK_EQUAL("default", opt3.get_value());
  }

  TEST(TestSetGetValue)
  {
    option opt(
        "s",
        "long",
        "Brief.",
        "Detailed.");
    option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        "default");
    option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        "default",
        "initial");

    CHECK(!opt.has_value());
    CHECK(opt2.has_value());
    CHECK(opt3.has_value());

    opt.set_value("value");
    opt2.set_value("value");
    opt3.set_value("value");

    CHECK_EQUAL("value", opt.get_value());
    CHECK_EQUAL("value", opt2.get_value());
    CHECK_EQUAL("value", opt3.get_value());

    opt.set_value();
    opt2.set_value();
    opt3.set_value();

    CHECK_THROW(opt.get_value(), no_value_error);
    CHECK_EQUAL("default", opt2.get_value());
    CHECK_EQUAL("default", opt3.get_value());
  }

  TEST(TestCreate)
  {
    option opt(
        "s",
        "long",
        "Brief.",
        "Detailed.");
    option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        "default");
    option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        "default",
        "initial");

    CHECK(!opt.has_value());
    CHECK(opt2.has_value());
    CHECK(opt3.has_value());

    opt.set_value("value");
    opt2.set_value("value");
    opt3.set_value("value");

    value const* val = opt.create();
    value const* val2 = opt2.create();
    value const* val3 = opt3.create();

    CHECK_EQUAL("value", val->get());
    CHECK_EQUAL("value", val2->get());
    CHECK_EQUAL("value", val3->get());

    opt.set_value();
    opt2.set_value();
    opt3.set_value();

    value const* valb2 = opt2.create();
    value const* valb3 = opt3.create();

    CHECK_THROW(opt.create(), no_value_error);
    CHECK_EQUAL("default", valb2->get());
    CHECK_EQUAL("default", valb3->get());

    delete val;
    delete val2;
    delete val3;
    delete valb2;
    delete valb3;
  }

  TEST(TestHasChildOptions)
  {
    const option opt(
        "s",
        "long",
        "Brief.",
        "Detailed.");
    const option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        "default");
    const option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        "default",
        "initial");
    CHECK(!opt.has_child_options());
    CHECK(!opt2.has_child_options());
    CHECK(!opt3.has_child_options());
  }

  TEST(TestFailGetChildOptions)
  {
    const option opt(
        "s",
        "long",
        "Brief.",
        "Detailed.");
    const option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        "default");
    const option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        "default",
        "initial");
    CHECK_THROW(opt.get_child_options(), no_such_namespace_error);
    CHECK_THROW(opt2.get_child_options(), no_such_namespace_error);
    CHECK_THROW(opt3.get_child_options(), no_such_namespace_error);
  }

  TEST(TestFailGetChildOption)
  {
    const option opt(
        "s",
        "long",
        "Brief.",
        "Detailed.");
    const option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        "default");
    const option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        "default",
        "initial");
    CHECK_THROW(opt.get_child_option("flag"), no_such_namespace_error);
    CHECK_THROW(opt2.get_child_option("lag"), no_such_namespace_error);
    CHECK_THROW(opt3.get_child_option("lag"), no_such_namespace_error);
  }

  TEST(TestIsSelection)
  {
    const option opt(
        "s",
        "long",
        "Brief.",
        "Detailed.");
    const option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        "default");
    const option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        "default",
        "initial");
    CHECK(!opt.is_selection());
    CHECK(!opt2.is_selection());
    CHECK(!opt3.is_selection());
  }

  TEST(TestSelectionIterator)
  {
    const option opt(
        "s",
        "long",
        "Brief.",
        "Detailed.");
    const option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        "default");
    const option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        "default",
        "initial");

    CHECK_THROW(opt.selection_values_begin(), no_selection_error);
    CHECK_THROW(opt2.selection_values_begin(), no_selection_error);
    CHECK_THROW(opt3.selection_values_begin(), no_selection_error);

    CHECK_THROW(opt.selection_values_end(), no_selection_error);
    CHECK_THROW(opt2.selection_values_end(), no_selection_error);
    CHECK_THROW(opt3.selection_values_end(), no_selection_error);
  }

  TEST(TestIsValidValue)
  {
    const option opt(
        "s",
        "long",
        "Brief.",
        "Detailed.");
    const option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        "default");
    const option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        "default",
        "initial");

    CHECK(opt.is_valid_value(""));
    CHECK(opt2.is_valid_value(""));
    CHECK(opt3.is_valid_value(""));

    CHECK(opt.is_valid_value("42"));
    CHECK(opt2.is_valid_value("42"));
    CHECK(opt3.is_valid_value("42"));

    CHECK(opt.is_valid_value("hello world\nlong sentence.\nAnd $%&/!(($))?"));
    CHECK(opt2.is_valid_value("hello world\nlong sentence.\nAnd $%&/!(($))?"));
    CHECK(opt3.is_valid_value("hello world\nlong sentence.\nAnd $%&/!(($))?"));
  }

}

