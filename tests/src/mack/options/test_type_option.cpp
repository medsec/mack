#include <unittest++/UnitTest++.h>

#include <mack/options/type_option.hpp>
#include <mack/options/exceptions.hpp>
#include <mack/core/null_pointer_error.hpp>

#include <string>

#include <mack/options/types/test_option_type.hpp>

#include <boost/exception/exception.hpp>
#include <boost/exception/diagnostic_information.hpp>

using namespace mack::options;

SUITE(mack_options_type_option)
{
  TEST(TestSimpleConstructor)
  {
    const option_type* type = new test_option_type;
    const type_option opt(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        type);

    CHECK(!opt.has_value());
  }

  TEST(TestDefaultValueConstructor)
  {
    const option_type* type = new test_option_type;
    const type_option opt(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        type,
        "tests::mack::options::ambiguous_class");
    CHECK(opt.has_value());
    CHECK_EQUAL("tests::mack::options::ambiguous_class", opt.get_value());
  }

  TEST(TestDefaultStartValueConstructor)
  {
    const option_type* type = new test_option_type;
    const type_option opt(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        type,
        "tests::mack::options::ambiguous_class",
        "tests::mack::options::recursive_class");

    CHECK(opt.has_value());
    CHECK_EQUAL("tests::mack::options::recursive_class", opt.get_value());
  }

  TEST(TestConstructorFailNullPointer)
  {
    CHECK_THROW(type_option opt1(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        NULL),
        mack::core::null_pointer_error);
    CHECK_THROW(type_option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        NULL,
        "tests::mack::options::ambiguous_class"),
        mack::core::null_pointer_error);
    CHECK_THROW(type_option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        NULL,
        "tests::mack::options::ambiguous_class",
        "tests::mack::options::recursive_class"),
         mack::core::null_pointer_error);
  }

  TEST(TestConstructorFailNoSuchValue)
  {
    CHECK_THROW(type_option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type,
        "tests::mack::options::ambiguous_clazz"),
        invalid_value_error);
    CHECK_THROW(type_option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type,
        "tests::mack::options::none::ambiguous_class",
        "tests::mack::options::recursive_class"),
         invalid_value_error);
    CHECK_THROW(type_option opt4(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type,
        "tests::mack::options::ambiguous_class",
        ""),
         invalid_value_error);
  }

  TEST(TestConstructorFailAmbiguousValue)
  {
    CHECK_THROW(type_option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type,
        "ambiguous_class"),
        ambiguous_value_error);
    CHECK_THROW(type_option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type,
        "ambiguous_class",
        "tests::mack::options::recursive_class"),
         ambiguous_value_error);
    CHECK_THROW(type_option opt4(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type,
        "tests::mack::options::ambiguous_class",
        "Ambiguous_Class"),
         ambiguous_value_error);
  }

  TEST(TestGetShortFlag)
  {
    type_option opt1(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type);
    type_option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type,
        "tests::mack::options::ambiguous_class");
    type_option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type,
        "tests::mack::options::ambiguous_class",
        "tests::mack::options::recursive_class");

    CHECK_EQUAL("s", opt1.get_short_flag());
    CHECK_EQUAL("s", opt2.get_short_flag());
    CHECK_EQUAL("s", opt3.get_short_flag());
  }

  TEST(TestGetLongFlag)
  {
    type_option opt1(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type);
    type_option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type,
        "tests::mack::options::ambiguous_class");
    type_option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type,
        "tests::mack::options::ambiguous_class",
        "tests::mack::options::recursive_class");
    CHECK_EQUAL("long", opt1.get_long_flag());
    CHECK_EQUAL("long", opt2.get_long_flag());
    CHECK_EQUAL("long", opt3.get_long_flag());
  }

  TEST(TestGetBrief)
  {
    type_option opt1(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type);
    type_option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type,
        "tests::mack::options::ambiguous_class");
    type_option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type,
        "tests::mack::options::ambiguous_class",
        "tests::mack::options::recursive_class");
    CHECK_EQUAL("Brief.", opt1.get_brief_description());
    CHECK_EQUAL("Brief.", opt2.get_brief_description());
    CHECK_EQUAL("Brief.", opt3.get_brief_description());
  }

  TEST(TestGetFullNoDetailed)
  {
    type_option opt1(
        "s",
        "long",
        "Brief.",
        "",
        false,
        new test_option_type);
    type_option opt2(
        "s",
        "long",
        "Brief.",
        "",
        false,
        new test_option_type,
        "tests::mack::options::ambiguous_class");
    type_option opt3(
        "s",
        "long",
        "Brief.",
        "",
        false,
        new test_option_type,
        "tests::mack::options::ambiguous_class",
        "tests::mack::options::recursive_class");
    CHECK_EQUAL("Brief.", opt1.get_full_description());
    CHECK_EQUAL("Brief.", opt2.get_full_description());
    CHECK_EQUAL("Brief.", opt3.get_full_description());
  }

  TEST(TestGetFull)
  {
    type_option opt1(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type);
    type_option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type,
        "tests::mack::options::ambiguous_class");
    type_option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type,
        "tests::mack::options::ambiguous_class",
        "tests::mack::options::recursive_class");
    CHECK_EQUAL("Brief.\nDetailed.", opt1.get_full_description());
    CHECK_EQUAL("Brief.\nDetailed.", opt2.get_full_description());
    CHECK_EQUAL("Brief.\nDetailed.", opt3.get_full_description());
  }

  TEST(TestSetGetValueEmpty)
  {
    type_option opt1(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type);
    type_option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type,
        "tests::mack::options::ambiguous_class");
    type_option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type,
        "tests::mack::options::ambiguous_class",
        "tests::mack::options::recursive_class");

    CHECK(!opt1.has_value());
    CHECK(opt2.has_value());
    CHECK(opt3.has_value());

    CHECK_THROW(opt1.get_value(), no_value_error);
    CHECK_EQUAL("tests::mack::options::ambiguous_class", opt2.get_value());
    CHECK_EQUAL("tests::mack::options::recursive_class", opt3.get_value());

    opt1.set_value();
    opt2.set_value();
    opt3.set_value();

    CHECK_THROW(opt1.get_value(), no_value_error);
    CHECK_EQUAL("tests::mack::options::ambiguous_class", opt2.get_value());
    CHECK_EQUAL("tests::mack::options::ambiguous_class", opt3.get_value());
  }

  TEST(TestSetGetValue)
  {
    type_option opt1(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type);
    type_option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type,
        "tests::mack::options::ambiguous_class");
    type_option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type,
        "tests::mack::options::ambiguous_class",
        "tests::mack::options::recursive_class");

    CHECK(!opt1.has_value());
    CHECK(opt2.has_value());
    CHECK(opt3.has_value());

    opt1.set_value("tests::mack::ambiguous_class");
    opt2.set_value("tests::mack::ambiguous_class");
    opt3.set_value("tests::mack::ambiguous_class");

    CHECK_EQUAL("tests::mack::ambiguous_class", opt1.get_value());
    CHECK_EQUAL("tests::mack::ambiguous_class", opt2.get_value());
    CHECK_EQUAL("tests::mack::ambiguous_class", opt3.get_value());

    opt1.set_value();
    opt2.set_value();
    opt3.set_value();

    CHECK_THROW(opt1.get_value(), no_value_error);
    CHECK_EQUAL("tests::mack::options::ambiguous_class", opt2.get_value());
    CHECK_EQUAL("tests::mack::options::ambiguous_class", opt3.get_value());
  }

  TEST(TestSetGetValueSimpleNames)
  {
    type_option opt(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type);

    CHECK(!opt.has_value());
    opt.set_value("tests::mack::options::recursive_class");
    CHECK_EQUAL("tests::mack::options::recursive_class", opt.get_value());

    opt.set_value();
    CHECK(!opt.has_value());
    opt.set_value("Tests::mack::OpTions::Recursive_cLass");
    CHECK_EQUAL("tests::mack::options::recursive_class", opt.get_value());

    opt.set_value();
    CHECK(!opt.has_value());
    opt.set_value("recursive_class");
    CHECK_EQUAL("tests::mack::options::recursive_class", opt.get_value());

    opt.set_value();
    CHECK(!opt.has_value());
    opt.set_value("ReCursive_ClasS");
    CHECK_EQUAL("tests::mack::options::recursive_class", opt.get_value());
  }

  TEST(TestCreate)
  {
    type_option opt1(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type);
    type_option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type,
        "tests::mack::options::ambiguous_class");
    type_option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type,
        "tests::mack::options::ambiguous_class",
        "tests::mack::options::recursive_class");

    opt1.set_value("tests::mack::options::empty_class");
    opt2.set_value("empty_class");
    opt3.set_value("Empty_Class");

    value const* val = opt1.create();
    value const* val2 = opt2.create();
    value const* val3 = opt3.create();

    delete val->get<tests::mack::base_class>();
    delete val2->get<tests::mack::base_class>();
    delete val3->get<tests::mack::base_class>();

    delete val;
    delete val2;
    delete val3;
  }

  TEST(TestCreateRecursive)
  {
    type_option opt(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type);

    CHECK(!opt.has_child_options());
    opt.set_value("tests::mack::options::recursive_class");
    CHECK(opt.has_child_options());

    option* opt_1 = opt.get_child_option("standard");
    option* opt_2 = opt.get_child_option("default");
    option* opt_3 = opt.get_child_option("initial");

    CHECK(!opt_1->has_value());
    CHECK(opt_2->has_value());
    CHECK(opt_3->has_value());

    CHECK_EQUAL("tests::mack::options::ambiguous_class", opt_2->get_value());
    CHECK_EQUAL("tests::mack::options::ambiguous_class", opt_3->get_value());

    opt_1->set_value("empty_class");
    opt_2->set_value("tests::mack::Ambiguous_Class");
    CHECK_EQUAL("tests::mack::options::empty_class", opt_1->get_value());
    CHECK_EQUAL("tests::mack::ambiguous_class", opt_2->get_value());
    CHECK_EQUAL("tests::mack::options::ambiguous_class", opt_3->get_value());

    CHECK(opt_1->has_child_options());
    CHECK(opt_1->get_child_options()->is_empty());

    CHECK(opt_2->has_child_options());
    option* opt_2_1  = opt_2->get_child_option("standard");
    option* opt_2_2  = opt_2->get_child_option("default");
    option* opt_2_3  = opt_2->get_child_option("initial");

    CHECK(!opt_2_1->has_value());
    opt_2_1->set_value("set value");
    CHECK(opt_2_1->has_value());
    CHECK(opt_2_2->has_value());
    CHECK(opt_2_3->has_value());
    CHECK_EQUAL("set value", opt_2_1->get_value());
    CHECK_EQUAL("default value", opt_2_2->get_value());
    CHECK_EQUAL("initial value", opt_2_3->get_value());

    CHECK(opt_3->has_child_options());
    option* opt_3_1  = opt_3->get_child_option("standard");
    option* opt_3_2  = opt_3->get_child_option("default");
    option* opt_3_3  = opt_3->get_child_option("initial");

    CHECK(!opt_3_1->has_value());
    opt_3_1->set_value("option 2");
    CHECK(opt_3_1->has_value());
    CHECK(opt_3_2->has_value());
    CHECK(opt_3_3->has_value());
    CHECK_EQUAL("option 2", opt_3_1->get_value());
    CHECK_EQUAL("option 1", opt_3_2->get_value());
    CHECK_EQUAL("option 1", opt_3_3->get_value());

    value const* val = opt.create();

    delete val->get<tests::mack::base_class>();

    delete val;
  }

  TEST(TestGetChildOptions)
  {
    type_option opt1(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type);
    type_option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type,
        "tests::mack::options::ambiguous_class");
    type_option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type,
        "tests::mack::options::ambiguous_class",
        "tests::mack::options::recursive_class");

    CHECK_THROW(opt1.get_child_options(), no_such_namespace_error);

    options* opts_2 = opt2.get_child_options();
    CHECK_EQUAL(
        std::string("Brief: Ambiguous Class.\n")
        + "Details: tests::mack::options::ambiguous_class",
        opts_2->get_full_description());

    options* opts_3 = opt3.get_child_options();
    CHECK_EQUAL(
        std::string("Brief: Recursive Class.\n")
        + "Details: tests::mack::options::recursive_class",
        opts_3->get_full_description());

    opt3.set_value("empty_class");
    opts_3 = opt3.get_child_options();
    CHECK_EQUAL(
        std::string("Brief: Empty Class.\n")
        + "Details: tests::mack::options::empty_class",
        opts_3->get_full_description());
  }

  TEST(TestIsSelection)
  {
    type_option opt1(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type);
    type_option opt2(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type,
        "tests::mack::options::ambiguous_class");
    type_option opt3(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type,
        "tests::mack::options::ambiguous_class",
        "tests::mack::options::recursive_class");
    CHECK(opt1.is_selection());
    CHECK(opt2.is_selection());
    CHECK(opt3.is_selection());
  }

  TEST(TestSelectionIterator)
  {
    type_option opt(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type);

    std::vector<std::string>::const_iterator it = opt.selection_values_begin();
    CHECK(it != opt.selection_values_end());
    CHECK_EQUAL("tests::mack::options::recursive_class", *it);
    it++;

    CHECK(it != opt.selection_values_end());
    CHECK_EQUAL("tests::mack::options::ambiguous_class", *it);
    it++;

    CHECK(it != opt.selection_values_end());
    CHECK_EQUAL("tests::mack::options::empty_class", *it);
    it++;

    CHECK(it != opt.selection_values_end());
    CHECK_EQUAL("tests::mack::ambiguous_class", *it);
    it++;

    CHECK(it == opt.selection_values_end());
  }

  TEST(TestIsValidValue)
  {
    type_option opt(
        "s",
        "long",
        "Brief.",
        "Detailed.",
        false,
        new test_option_type);

    CHECK(!opt.is_valid_value(""));
    CHECK(!opt.is_valid_value("42"));

    CHECK(opt.is_valid_value("tests::mack::options::recursive_class"));
    CHECK(opt.is_valid_value("tests::mack::options::ambiguous_class"));
    CHECK(opt.is_valid_value("tests::mack::options::empty_class"));
    CHECK(opt.is_valid_value("tests::mack::ambiguous_class"));

    CHECK(opt.is_valid_value("tests::mack::options::Recursive_class"));
    CHECK(opt.is_valid_value("tests::mack::options::ambIguous_class"));
    CHECK(opt.is_valid_value("tests::mack::options::emptY_Class"));
    CHECK(opt.is_valid_value("TESTS::MACK::ambiguous_class"));

    CHECK(opt.is_valid_value("recursive_class"));
    CHECK(opt.is_valid_value("empty_class"));
    CHECK(!opt.is_valid_value("ambiguous_class"));

    CHECK(opt.is_valid_value("recuRsiVe_class"));
    CHECK(opt.is_valid_value("EMPTY_CLASS"));
    CHECK(!opt.is_valid_value("AmbiguouS_ClasS"));
  }
}

