#include <unittest++/UnitTest++.h>

#include <mack/options/value.hpp>

#include <mack/options/exceptions.hpp>

#include <string>

using namespace mack::options;

SUITE(mack_options_value)
{
  TEST(TestString)
  {
    const std::string str("value");
    const value v(str);
    CHECK_EQUAL(str, v.get());
  }

  TEST(TestCharArray)
  {
    const char* str_array = "value";
    const value v(str_array);
    CHECK_EQUAL(std::string(str_array), v.get());
  }

  TEST(TestPointer)
  {
    const std::string* str_ptr = new std::string("value");
    const value v(str_ptr, "std::string");
    CHECK_EQUAL(*str_ptr, *v.get<const std::string>());
    delete str_ptr;
  }

  TEST(TestModifyPointer)
  {
    std::string* str_ptr = new std::string("value");
    const value v(str_ptr, "std::string");

    str_ptr->append("s");
    CHECK_EQUAL(*str_ptr, *v.get<std::string>());
    delete str_ptr;
  }

  TEST(TestObjectNotDeleted)
  {
    std::string* str_ptr = new std::string("value");
    {
      const value v(str_ptr, "std::string");
    } // scope ended -> v is deleted
    CHECK_EQUAL("value", *str_ptr);
    delete str_ptr;
  }

  TEST(TestCast)
  {
    const value value_int("42");
    const value value_float("4.2");
    const value value_bool("1");

    CHECK_EQUAL(42, value_int.cast<int>());
    CHECK_CLOSE(4.2f, value_float.cast<float>(), 0.00001);
    CHECK(value_bool.cast<bool>());
  }

  TEST(TestGetValueClassName)
  {
    std::string* str_ptr = new std::string("value");
    const value v(str_ptr, "std::string");

    CHECK_EQUAL("std::string", v.get_value_class_name());
    delete str_ptr;
  }

  TEST(TestFailPointerToString)
  {
    const std::string* str_ptr = new std::string("42");
    const value val(str_ptr, "std::string");
    CHECK_THROW(val.get(), boost::bad_any_cast);
    CHECK_THROW(val.cast<int>(), boost::bad_any_cast);
    delete str_ptr;
  }

  TEST(TestFailPointerToDifferentPointer)
  {
    const int* int_ptr = new int(42);
    const value val(int_ptr, "int");
    CHECK_THROW(val.get<std::string>(), boost::bad_any_cast);
    delete int_ptr;
  }

  TEST(TestFailStringToPointer)
  {
    std::string str("42");
    const value val(str);
    CHECK_THROW(val.get<std::string>(), boost::bad_any_cast);
  }

  TEST(TestFailGetValueClassName)
  {
    std::string str("42");
    const value val(str);
    CHECK_THROW(val.get_value_class_name(), invalid_type_error);
  }

  TEST(TestFailCast)
  {
    const int* int_ptr = new int(42);
    const value value_ptr(int_ptr, "int");
    const value value_int("42");
    const value value_float("4.2");
    const value value_string("E42");

    CHECK_THROW(value_ptr.cast<int>(), boost::bad_any_cast);
    CHECK_THROW(value_int.cast<bool>(), boost::bad_lexical_cast);
    CHECK_THROW(value_float.cast<int>(), boost::bad_lexical_cast);
    CHECK_THROW(value_string.cast<float>(), boost::bad_lexical_cast);
    delete int_ptr;
  }
}

