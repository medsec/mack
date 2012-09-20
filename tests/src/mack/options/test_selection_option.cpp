#include <unittest++/UnitTest++.h>

#include <mack/options/selection_option.hpp>
#include <mack/options/exceptions.hpp>

#include <string>

using namespace mack::options;

SUITE(mack_options_selection_option)
{
  TEST(TestConstructor)
  {
    std::vector<std::string> values;
    values.push_back("default");
    values.push_back("initial");
    values.push_back("other");

    selection_option opt1(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values);
    selection_option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values,
        "default");
    selection_option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values,
        "default",
        "initial");

    CHECK(!opt1.has_value());
    CHECK(opt2.has_value());
    CHECK(opt3.has_value());
    CHECK_EQUAL("default", opt2.get_value());
    CHECK_EQUAL("initial", opt3.get_value());
  }

  TEST(TestConstructorFailNoSelection)
  {
    std::vector<std::string> values;

    CHECK_THROW(selection_option opt1(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values), no_selection_error);
    CHECK_THROW(selection_option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values,
        "default"), no_selection_error);
    CHECK_THROW(selection_option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values,
        "default",
        "initial"), no_selection_error);
  }

  TEST(TestConstructorFailInvalidValue)
  {
    std::vector<std::string> values;
    values.push_back("other");

    CHECK_THROW(selection_option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values,
        "default"), invalid_value_error);
    CHECK_THROW(selection_option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values,
        "default",
        "initial"), invalid_value_error);
  }

  TEST(TestSetValue)
  {
    std::vector<std::string> values;
    values.push_back("default");
    values.push_back("initial");
    values.push_back("other");

    selection_option opt1(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values);
    selection_option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values,
        "default");
    selection_option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values,
        "default",
        "initial");

    opt1.set_value("other");
    opt2.set_value("other");
    opt3.set_value("other");
    CHECK(opt1.has_value());
    CHECK(opt2.has_value());
    CHECK(opt3.has_value());
    CHECK_EQUAL("other", opt1.get_value());
    CHECK_EQUAL("other", opt2.get_value());
    CHECK_EQUAL("other", opt3.get_value());

    opt1.set_value();
    opt2.set_value();
    opt3.set_value();
    CHECK(!opt1.has_value());
    CHECK(opt2.has_value());
    CHECK(opt3.has_value());
    CHECK_EQUAL("default", opt2.get_value());
    CHECK_EQUAL("default", opt3.get_value());
  }

  TEST(TestSetValueFailInvalidValue)
  {
    std::vector<std::string> values;
    values.push_back("default");
    values.push_back("initial");

    selection_option opt1(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values);
    selection_option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values,
        "default");
    selection_option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values,
        "default",
        "initial");

    CHECK_THROW(opt1.set_value("other"), invalid_value_error);
    CHECK_THROW(opt2.set_value("other"), invalid_value_error);
    CHECK_THROW(opt3.set_value("other"), invalid_value_error);
  }

  TEST(TestIsSelection)
  {
    std::vector<std::string> values;
    values.push_back("default");
    values.push_back("initial");
    values.push_back("other");

    selection_option opt1(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values);
    selection_option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values,
        "default");
    selection_option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values,
        "default",
        "initial");

    CHECK(opt1.is_selection());
    CHECK(opt2.is_selection());
    CHECK(opt3.is_selection());
  }

  TEST(TestSelectionIterator)
  {
    std::vector<std::string> values;
    values.push_back("default");
    values.push_back("initial");

    selection_option opt1(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values);
    selection_option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values,
        "default");
    selection_option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values,
        "default",
        "initial");

    std::vector<std::string>::const_iterator it;

    it = opt1.selection_values_begin();
    CHECK(it != opt1.selection_values_end());
    CHECK_EQUAL(*it, "default");
    it++;
    CHECK(it != opt1.selection_values_end());
    CHECK_EQUAL(*it, "initial");
    it++;
    CHECK(it == opt1.selection_values_end());

    it = opt2.selection_values_begin();
    CHECK(it != opt2.selection_values_end());
    CHECK_EQUAL(*it, "default");
    it++;
    CHECK(it != opt2.selection_values_end());
    CHECK_EQUAL(*it, "initial");
    it++;
    CHECK(it == opt2.selection_values_end());

    it = opt3.selection_values_begin();
    CHECK(it != opt3.selection_values_end());
    CHECK_EQUAL(*it, "default");
    it++;
    CHECK(it != opt3.selection_values_end());
    CHECK_EQUAL(*it, "initial");
    it++;
    CHECK(it == opt3.selection_values_end());
  }

  TEST(TestGetShortFlag)
  {
    std::vector<std::string> values;
    values.push_back("default");
    values.push_back("initial");

    selection_option opt1(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values);
    selection_option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values,
        "default");
    selection_option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values,
        "default",
        "initial");

    CHECK_EQUAL("s", opt1.get_short_flag());
    CHECK_EQUAL("s", opt2.get_short_flag());
    CHECK_EQUAL("s", opt3.get_short_flag());
  }

  TEST(TestGetLongFlag)
  {
    std::vector<std::string> values;
    values.push_back("default");
    values.push_back("initial");

    selection_option opt1(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values);
    selection_option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values,
        "default");
    selection_option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values,
        "default",
        "initial");

    CHECK_EQUAL("long", opt1.get_long_flag());
    CHECK_EQUAL("long", opt2.get_long_flag());
    CHECK_EQUAL("long", opt3.get_long_flag());
  }

  TEST(TestGetBrief)
  {
    std::vector<std::string> values;
    values.push_back("default");
    values.push_back("initial");

    selection_option opt1(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values);
    selection_option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values,
        "default");
    selection_option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values,
        "default",
        "initial");

    CHECK_EQUAL("Brief.", opt1.get_brief_description());
    CHECK_EQUAL("Brief.", opt2.get_brief_description());
    CHECK_EQUAL("Brief.", opt3.get_brief_description());
  }

  TEST(TestGetFullNoDetailed)
  {
    std::vector<std::string> values;
    values.push_back("default");
    values.push_back("initial");

    selection_option opt1(
        "s",
        "long",
        "Brief.",
        "",
        values);
    selection_option opt2(
        "s",
        "long",
        "Brief.",
        "",
        values,
        "default");
    selection_option opt3(
        "s",
        "long",
        "Brief.",
        "",
        values,
        "default",
        "initial");

    CHECK_EQUAL("Brief.", opt1.get_full_description());
    CHECK_EQUAL("Brief.", opt2.get_full_description());
    CHECK_EQUAL("Brief.", opt3.get_full_description());
  }

  TEST(TestGetFull)
  {
    std::vector<std::string> values;
    values.push_back("default");
    values.push_back("initial");

    selection_option opt1(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values);
    selection_option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values,
        "default");
    selection_option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values,
        "default",
        "initial");

    CHECK_EQUAL("Brief.\nDetailed.", opt1.get_full_description());
    CHECK_EQUAL("Brief.\nDetailed.", opt2.get_full_description());
    CHECK_EQUAL("Brief.\nDetailed.", opt3.get_full_description());
  }

  TEST(TestCreate)
  {
    std::vector<std::string> values;
    values.push_back("default");
    values.push_back("initial");
    values.push_back("other");

    selection_option opt1(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values);
    selection_option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values,
        "default");
    selection_option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values,
        "default",
        "initial");

    CHECK(!opt1.has_value());
    CHECK(opt2.has_value());
    CHECK(opt3.has_value());

    opt1.set_value("other");

    value const* val = opt1.create();
    value const* val2 = opt2.create();
    value const* val3 = opt3.create();

    CHECK_EQUAL("other", val->get());
    CHECK_EQUAL("default", val2->get());
    CHECK_EQUAL("initial", val3->get());

    opt1.set_value();
    opt2.set_value();
    opt3.set_value();

    value const* valb2 = opt2.create();
    value const* valb3 = opt3.create();

    CHECK_THROW(opt1.create(), no_value_error);
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
    std::vector<std::string> values;
    values.push_back("default");
    values.push_back("initial");

    selection_option opt1(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values);
    selection_option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values,
        "default");
    selection_option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values,
        "default",
        "initial");

    CHECK(!opt1.has_child_options());
    CHECK(!opt2.has_child_options());
    CHECK(!opt3.has_child_options());
  }

  TEST(TestFailGetChildOptions)
  {
    std::vector<std::string> values;
    values.push_back("default");
    values.push_back("initial");

    selection_option opt1(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values);
    selection_option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values,
        "default");
    selection_option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values,
        "default",
        "initial");
    CHECK_THROW(opt1.get_child_options(), no_such_namespace_error);
    CHECK_THROW(opt2.get_child_options(), no_such_namespace_error);
    CHECK_THROW(opt3.get_child_options(), no_such_namespace_error);
  }

  TEST(TestFailGetChildOption)
  {
    std::vector<std::string> values;
    values.push_back("default");
    values.push_back("initial");

    selection_option opt1(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values);
    selection_option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values,
        "default");
    selection_option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values,
        "default",
        "initial");
    CHECK_THROW(opt1.get_child_option("flag"), no_such_namespace_error);
    CHECK_THROW(opt2.get_child_option("lag"), no_such_namespace_error);
    CHECK_THROW(opt3.get_child_option("lag"), no_such_namespace_error);
  }

  TEST(TestIsValidValue)
  {
    std::vector<std::string> values;
    values.push_back("default");
    values.push_back("initial");
    values.push_back("other");

    selection_option opt1(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values);
    selection_option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values,
        "default");
    selection_option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        values,
        "default",
        "initial");

    values.push_back("none");

    CHECK(!opt1.is_valid_value(""));
    CHECK(!opt2.is_valid_value(""));
    CHECK(!opt3.is_valid_value(""));

    CHECK(opt1.is_valid_value("default"));
    CHECK(opt2.is_valid_value("default"));
    CHECK(opt3.is_valid_value("default"));

    CHECK(opt1.is_valid_value("initial"));
    CHECK(opt2.is_valid_value("initial"));
    CHECK(opt3.is_valid_value("initial"));

    CHECK(opt1.is_valid_value("other"));
    CHECK(opt2.is_valid_value("other"));
    CHECK(opt3.is_valid_value("other"));

    CHECK(!opt1.is_valid_value("none"));
    CHECK(!opt2.is_valid_value("none"));
    CHECK(!opt3.is_valid_value("none"));
  }
}

