#include <unittest++/UnitTest++.h>

#include <mack/options/commandline_formatter.hpp>

#include <mack/options/exceptions.hpp>

#include <mack/options/selection_option.hpp>
#include <mack/options/type_option.hpp>
#include <mack/options/types/test_option_type.hpp>

#include <string>
#include <sstream>

using namespace mack::options;

SUITE(mack_options_commandline_formatter)
{
  TEST(PrintProgramDescription)
  {
    std::ostringstream oss;
    commandline_formatter formatter(oss, true, false, 40, 2);

    formatter.print_description(
        "MyProgram",
        "A brief description.",
        std::string("A brief description.\n")
        + "A very long description which is much longer than "
        + "forty characters.\n"
        + "And contains a linebreak.");

    CHECK_EQUAL(
        std::string("MyProgram\n\n")
        + "A brief description.\n",
        oss.str());

    std::ostringstream oss2;
    commandline_formatter formatter2(oss2, true, true, 40, 2);

    formatter2.print_description(
        "MyProgram",
        "A brief description.",
        std::string("A brief description.\n")
        + "A very long description which is much longer than "
        + "forty characters.\n"
        + "And contains a linebreak.");

    CHECK_EQUAL(
        std::string("MyProgram\n\n")
        + "A brief description.\n"
        + "A very long description which is much\n"
        + "longer than forty characters.\n"
        + "And contains a linebreak.\n",
        oss2.str());

    std::ostringstream oss3;
    commandline_formatter formatter3(oss3, true, true, 10, 2);

    formatter3.print_description(
        "MyProgram",
        "A brief description.",
        std::string("A brief description.\n")
        + "A very long description which is much longer than "
        + "forty characters.\n"
        + "And contains a linebreak.");

    CHECK_EQUAL(
        std::string("MyProgram\n\n")
        + "A brief\n"
        + "description.\n"
        + "A very\n"
        + "long\n"
        + "description\n"
        + "which is\n"
        + "much\n"
        + "longer\n"
        + "than forty\n"
        + "characters.\n"
        + "And\n"
        + "contains a\n"
        + "linebreak.\n",
        oss3.str());
  }

  TEST(PrintOptionDescription)
  {
    option* opt = new option(
        "s",
        "long",
        "A brief description.",
        std::string("A very long description which is much longer than ")
        + "forty characters.\n"
        + "And contains a linebreak.");

    opt->set_value("value");

    std::ostringstream oss;
    commandline_formatter formatter(oss, true, false, 40, 2);
    formatter.print_description(opt);

    CHECK_EQUAL(
        std::string("A brief description.\n"),
        oss.str());

    std::ostringstream oss2;
    commandline_formatter formatter2(oss2, true, true, 40, 2);
    formatter2.print_description(opt);

    CHECK_EQUAL(
        std::string("A brief description.\n")
        + "A very long description which is much\n"
        + "longer than forty characters.\n"
        + "And contains a linebreak.\n",
        oss2.str());

    delete opt;
  }

  TEST(PrintOption)
  {
    option* opt = new option(
        "s",
        "long",
        "A brief description.",
        std::string("A very long description which is much longer than ")
        + "forty characters.\n"
        + "And contains a linebreak.");

    option* opt2 = new option(
        "s",
        "long",
        "A brief description.",
        std::string("A very long description which is much longer than ")
        + "forty characters.\n"
        + "And contains a linebreak.",
        "default option");

    std::ostringstream oss;
    commandline_formatter formatter(oss, true, false, 40, 2);
    formatter.print(opt);

    CHECK_EQUAL(
        std::string("-s, --long\n")
          + "    value  : MISSING\n"
          + "    A brief description.\n",
        oss.str());

    std::ostringstream oss2;
    commandline_formatter formatter2(oss2, false, false, 40, 2);
    formatter2.print(opt);

    CHECK_EQUAL(
        std::string("s, long\n")
          + "    value  : MISSING\n"
          + "    A brief description.\n",
        oss2.str());

    std::ostringstream oss3;
    commandline_formatter formatter3(oss3, true, true, 40, 2);
    formatter3.print(opt2);

    CHECK_EQUAL(
        std::string("-s, --long\n")
          + "    value  : default option\n"
          + "    default: default option\n"
          + "    A brief description.\n"
          + "    A very long description which is\n"
          + "    much longer than forty characters.\n"
          + "    And contains a linebreak.\n",
        oss3.str());

    opt2->set_value("a very long value for great good");
    std::ostringstream oss4;
    commandline_formatter formatter4(oss4, true, true, 40, 4);
    formatter4.print(opt2);

    CHECK_EQUAL(
        std::string("-s, --long\n")
          + "        value  : a very long value for\n"
          + "            great good\n"
          + "        default: default option\n"
          + "        A brief description.\n"
          + "        A very long description which is\n"
          + "        much longer than forty\n"
          + "        characters.\n"
          + "        And contains a linebreak.\n",
        oss4.str());

    delete opt;
    delete opt2;
  }

  TEST(PrintOptionNamespace)
  {
    option* opt = new option(
        "s",
        "long",
        "A brief description.",
        std::string("A very long description which is much longer than ")
        + "forty characters.\n"
        + "And contains a linebreak.");

    std::ostringstream oss;
    commandline_formatter formatter(oss, true, true, 40, 2);
    formatter.print(opt, "my.very.long.and.cool.name.space");

    CHECK_EQUAL(
        std::string("-s, --my.very.long.and.cool.name.space.long\n")
          + "    value  : MISSING\n"
          + "    A brief description.\n"
          + "    A very long description which is\n"
          + "    much longer than forty characters.\n"
          + "    And contains a linebreak.\n",
        oss.str());

    delete opt;
  }

  TEST(PrintSelectionOption)
  {
    std::vector<std::string> values;
    values.push_back("default option");
    values.push_back("a not so long value");
    values.push_back("a very long value for great good");

    option* opt = new selection_option(
        "s",
        "long",
        "A brief description.",
        std::string("A very long description which is much longer than ")
        + "forty characters.\n"
        + "And contains a linebreak.",
        values,
        "default option");

    opt->set_value("a very long value for great good");
    std::ostringstream oss;
    commandline_formatter formatter(oss, true, true, 40, 4);
    formatter.print(opt);
    CHECK_EQUAL(
        std::string("-s, --long\n")
          + "        value  : a very long value for\n"
          + "            great good\n"
          + "        default: default option\n"
          + "        possible values:\n"
          + "            - default option\n"
          + "            - a not so long value\n"
          + "            - a very long value for great good\n"
          + "        A brief description.\n"
          + "        A very long description which is\n"
          + "        much longer than forty\n"
          + "        characters.\n"
          + "        And contains a linebreak.\n",
        oss.str());

    delete opt;
  }

  TEST(PrintOptionsDescription)
  {
    option* opt = new type_option(
        "s",
        "long",
        "A brief description.",
        std::string("A very long description which is much longer than ")
        + "forty characters.\n"
        + "And contains a linebreak.",
        false,
        new test_option_type,
        "recursive_class",
        "EMPTY_CLASS");

    opt->set_value();
    opt->get_child_option("initial")->set_value("empty_class");

    std::ostringstream oss;
    commandline_formatter formatter(oss, true, true, 40, 4);
    formatter.print_description(opt->get_child_options());

    CHECK_EQUAL(
        std::string()
          + "Brief: Recursive Class.\n"
          + "Details:\n"
          + "tests::mack::options::recursive_class\n",
        oss.str());

    delete opt;
  }

  TEST(PrintOptions)
  {
    option* opt = new type_option(
        "s",
        "long",
        "A brief description.",
        std::string("A very long description which is much longer than ")
        + "forty characters.\n"
        + "And contains a linebreak.",
        false,
        new test_option_type,
        "recursive_class",
        "EMPTY_CLASS");

    opt->set_value();
    opt->get_child_option("initial")->set_value("empty_class");

    std::ostringstream oss;
    commandline_formatter formatter(oss, true, true, 40, 4);
    formatter.print(opt->get_child_options(), "my.namespace.long");

    CHECK_EQUAL(
        std::string()
          + "Brief: Recursive Class.\n"
          + "Details:\n"
          + "tests::mack::options::recursive_class\n"
          + "\n"
          + "-s, --my.namespace.long.standard\n"
          + "        value  : MISSING\n"
          + "        possible values:\n"
          + "            - tests::mack::options::recursive_class\n"
          + "            - tests::mack::options::ambiguous_class\n"
          + "            - tests::mack::options::empty_class\n"
          + "            - tests::mack::ambiguous_class\n"
          + "        A brief description: standard\n"
          + "        type.\n"
          + "        A detailed description: standard\n"
          + "        type.\n"
          + "\n"
          + "-d, --my.namespace.long.default\n"
          + "        value  : \n"
          + "            tests::mack::options::ambiguous_class\n"
          + "        default: \n"
          + "            tests::mack::options::ambiguous_class\n"
          + "        possible values:\n"
          + "            - tests::mack::options::recursive_class\n"
          + "            - tests::mack::options::ambiguous_class\n"
          + "            - tests::mack::options::empty_class\n"
          + "            - tests::mack::ambiguous_class\n"
          + "        A brief description: default\n"
          + "        type.\n"
          + "        A detailed description: default\n"
          + "        type.\n"
          + "    Brief: Ambiguous Class.\n"
          + "    Details:\n"
          + "    tests::mack::options::ambiguous_class\n"
          + "\n"
          + "    -s, --my.namespace.long.default.standard\n"
          + "            value  : MISSING\n"
          + "            possible values:\n"
          + "                - option 1\n"
          + "                - option 2\n"
          + "                - option 3\n"
          + "            A brief description:\n"
          + "            standard selection.\n"
          + "            A detailed description:\n"
          + "            standard selection.\n"
          + "\n"
          + "    -d, --my.namespace.long.default.default\n"
          + "            value  : option 1\n"
          + "            default: option 1\n"
          + "            possible values:\n"
          + "                - option 1\n"
          + "                - option 2\n"
          + "                - option 3\n"
          + "            A brief description: default\n"
          + "            selection.\n"
          + "            A detailed description:\n"
          + "            default selection.\n"
          + "\n"
          + "    -i, --my.namespace.long.default.initial\n"
          + "            value  : option 1\n"
          + "            default: option 2\n"
          + "            possible values:\n"
          + "                - option 1\n"
          + "                - option 2\n"
          + "                - option 3\n"
          + "            A brief description: initial\n"
          + "            selection.\n"
          + "            A detailed description:\n"
          + "            initial selection.\n"
          + "\n"
          + "    -b, --my.namespace.long.default.boolean\n"
          + "            value  : false\n"
          + "            default: true\n"
          + "            possible values:\n"
          + "                - true\n"
          + "                - false\n"
          + "            A brief description: switch.\n"
          + "            A detailed description:\n"
          + "            switch.\n"
          + "\n"
          + "-i, --my.namespace.long.initial\n"
          + "        value  : \n"
          + "            tests::mack::options::empty_class\n"
          + "        default: \n"
          + "            tests::mack::options::recursive_class\n"
          + "        possible values:\n"
          + "            - tests::mack::options::recursive_class\n"
          + "            - tests::mack::options::ambiguous_class\n"
          + "            - tests::mack::options::empty_class\n"
          + "            - tests::mack::ambiguous_class\n"
          + "        A brief description: initial\n"
          + "        type.\n"
          + "        A detailed description: initial\n"
          + "        type.\n"
          + "    Brief: Empty Class.\n"
          + "    Details:\n"
          + "    tests::mack::options::empty_class\n",
        oss.str());

    delete opt;
  }

  TEST(PrintTypeOption)
  {
    option* opt = new type_option(
        "s",
        "long",
        "A brief description.",
        std::string("A very long description which is much longer than ")
        + "forty characters.\n"
        + "And contains a linebreak.",
        false,
        new test_option_type,
        "recursive_class",
        "EMPTY_CLASS");

    std::ostringstream oss;
    commandline_formatter formatter(oss, true, true, 40, 4);
    formatter.print(opt);

    CHECK_EQUAL(
        std::string("-s, --long\n")
          + "        value  : \n"
          + "            tests::mack::options::empty_class\n"
          + "        default: \n"
          + "            tests::mack::options::recursive_class\n"
          + "        possible values:\n"
          + "            - tests::mack::options::recursive_class\n"
          + "            - tests::mack::options::ambiguous_class\n"
          + "            - tests::mack::options::empty_class\n"
          + "            - tests::mack::ambiguous_class\n"
          + "        A brief description.\n"
          + "        A very long description which is\n"
          + "        much longer than forty\n"
          + "        characters.\n"
          + "        And contains a linebreak.\n"
          + "    Brief: Empty Class.\n"
          + "    Details:\n"
          + "    tests::mack::options::empty_class\n",
        oss.str());

    opt->set_value();
    opt->get_child_option("initial")->set_value("empty_class");
    formatter.print(opt, "my.namespace");

    CHECK_EQUAL(
        std::string("-s, --long\n")
          + "        value  : \n"
          + "            tests::mack::options::empty_class\n"
          + "        default: \n"
          + "            tests::mack::options::recursive_class\n"
          + "        possible values:\n"
          + "            - tests::mack::options::recursive_class\n"
          + "            - tests::mack::options::ambiguous_class\n"
          + "            - tests::mack::options::empty_class\n"
          + "            - tests::mack::ambiguous_class\n"
          + "        A brief description.\n"
          + "        A very long description which is\n"
          + "        much longer than forty\n"
          + "        characters.\n"
          + "        And contains a linebreak.\n"
          + "    Brief: Empty Class.\n"
          + "    Details:\n"
          + "    tests::mack::options::empty_class\n"
          + "-s, --my.namespace.long\n"
          + "        value  : \n"
          + "            tests::mack::options::recursive_class\n"
          + "        default: \n"
          + "            tests::mack::options::recursive_class\n"
          + "        possible values:\n"
          + "            - tests::mack::options::recursive_class\n"
          + "            - tests::mack::options::ambiguous_class\n"
          + "            - tests::mack::options::empty_class\n"
          + "            - tests::mack::ambiguous_class\n"
          + "        A brief description.\n"
          + "        A very long description which is\n"
          + "        much longer than forty\n"
          + "        characters.\n"
          + "        And contains a linebreak.\n"
          + "    Brief: Recursive Class.\n"
          + "    Details:\n"
          + "    tests::mack::options::recursive_class\n"
          + "\n"
          + "    -s, --my.namespace.long.standard\n"
          + "            value  : MISSING\n"
          + "            possible values:\n"
          + "                - tests::mack::options::recursive_class\n"
          + "                - tests::mack::options::ambiguous_class\n"
          + "                - tests::mack::options::empty_class\n"
          + "                - tests::mack::ambiguous_class\n"
          + "            A brief description:\n"
          + "            standard type.\n"
          + "            A detailed description:\n"
          + "            standard type.\n"
          + "\n"
          + "    -d, --my.namespace.long.default\n"
          + "            value  : \n"
          + "                tests::mack::options::ambiguous_class\n"
          + "            default: \n"
          + "                tests::mack::options::ambiguous_class\n"
          + "            possible values:\n"
          + "                - tests::mack::options::recursive_class\n"
          + "                - tests::mack::options::ambiguous_class\n"
          + "                - tests::mack::options::empty_class\n"
          + "                - tests::mack::ambiguous_class\n"
          + "            A brief description: default\n"
          + "            type.\n"
          + "            A detailed description:\n"
          + "            default type.\n"
          + "        Brief: Ambiguous Class.\n"
          + "        Details:\n"
          + "        tests::mack::options::ambiguous_class\n"
          + "\n"
          + "        -s, --my.namespace.long.default.standard\n"
          + "                value  : MISSING\n"
          + "                possible values:\n"
          + "                    - option 1\n"
          + "                    - option 2\n"
          + "                    - option 3\n"
          + "                A brief description:\n"
          + "                standard selection.\n"
          + "                A detailed description:\n"
          + "                standard selection.\n"
          + "\n"
          + "        -d, --my.namespace.long.default.default\n"
          + "                value  : option 1\n"
          + "                default: option 1\n"
          + "                possible values:\n"
          + "                    - option 1\n"
          + "                    - option 2\n"
          + "                    - option 3\n"
          + "                A brief description:\n"
          + "                default selection.\n"
          + "                A detailed description:\n"
          + "                default selection.\n"
          + "\n"
          + "        -i, --my.namespace.long.default.initial\n"
          + "                value  : option 1\n"
          + "                default: option 2\n"
          + "                possible values:\n"
          + "                    - option 1\n"
          + "                    - option 2\n"
          + "                    - option 3\n"
          + "                A brief description:\n"
          + "                initial selection.\n"
          + "                A detailed description:\n"
          + "                initial selection.\n"
          + "\n"
          + "        -b, --my.namespace.long.default.boolean\n"
          + "                value  : false\n"
          + "                default: true\n"
          + "                possible values:\n"
          + "                    - true\n"
          + "                    - false\n"
          + "                A brief description:\n"
          + "                switch.\n"
          + "                A detailed description:\n"
          + "                switch.\n"
          + "\n"
          + "    -i, --my.namespace.long.initial\n"
          + "            value  : \n"
          + "                tests::mack::options::empty_class\n"
          + "            default: \n"
          + "                tests::mack::options::recursive_class\n"
          + "            possible values:\n"
          + "                - tests::mack::options::recursive_class\n"
          + "                - tests::mack::options::ambiguous_class\n"
          + "                - tests::mack::options::empty_class\n"
          + "                - tests::mack::ambiguous_class\n"
          + "            A brief description: initial\n"
          + "            type.\n"
          + "            A detailed description:\n"
          + "            initial type.\n"
          + "        Brief: Empty Class.\n"
          + "        Details:\n"
          + "        tests::mack::options::empty_class\n",
        oss.str());

    delete opt;
  }
}

