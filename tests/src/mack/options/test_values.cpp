#include <unittest++/UnitTest++.h>

#include <mack/options/values.hpp>
#include <mack/options/value.hpp>
#include <mack/core/null_pointer_error.hpp>

#include <string>
#include <map>

using namespace mack::options;

SUITE(mack_options_values)
{
  TEST(TestEmptyConstructor)
  {
    const values vals;
    CHECK(vals.is_empty());
  }

  TEST(TestMapConstructor)
  {
    std::map<std::string, value const*> map;

    const values vals_empty(map);

    value* val = new value("value");
    map.insert(std::pair<std::string, value const*>("flag", val));
    const values vals(map);

    CHECK(vals_empty.is_empty());
    CHECK(!vals.is_empty());
  }

  TEST(TestMapConstructorFailNull)
  {
    std::map<std::string, value const*> map;

    const values vals_empty(map);

    map.insert(std::pair<std::string, value const*>("flag", NULL));
    CHECK_THROW(const values vals(map), mack::core::null_pointer_error);
  }

  TEST(TestGetString)
  {
    std::map<std::string, value const*> map;
    value* val = new value("value");
    map.insert(std::pair<std::string, value const*>("flag", val));
    const values vals(map);

    CHECK_EQUAL(std::string("value"), vals.get("flag"));
  }

  TEST(TestGetBoolean)
  {
    std::map<std::string, value const*> map;
    map.insert(std::pair<std::string, value const*>("t_1", new value("true")));
    map.insert(std::pair<std::string, value const*>("t_2", new value("TRUE")));
    map.insert(std::pair<std::string, value const*>("t_3", new value("1")));
    map.insert(std::pair<std::string, value const*>("f_1", new value("false")));
    map.insert(std::pair<std::string, value const*>("f_2", new value("FALSE")));
    map.insert(std::pair<std::string, value const*>("f_3", new value("0")));
    const values vals(map);

    CHECK(vals.get_boolean("t_1"));
    CHECK(vals.get_boolean("t_2"));
    CHECK(vals.get_boolean("t_3"));
    CHECK(!vals.get_boolean("f_1"));
    CHECK(!vals.get_boolean("f_2"));
    CHECK(!vals.get_boolean("f_3"));
  }

  TEST(TestGetPointer)
  {
    std::string* str_ptr = new std::string("value");
    std::string* str_ptr_empty = new std::string();

    value* val = new value(str_ptr, "std::string");
    value* val_empty = new value(str_ptr_empty, "std::string");
    std::map<std::string, value const*> map;
    map.insert(std::pair<std::string, value const*>("flag", val));
    map.insert(std::pair<std::string, value const*>("flag2", val_empty));
    const values vals(map);

    CHECK_EQUAL(*str_ptr, *vals.get<std::string>("flag"));
    CHECK_EQUAL(*str_ptr_empty, *vals.get<std::string>("flag2"));
    delete str_ptr;
    delete str_ptr_empty;
  }

  TEST(TestGetValueClassName)
  {
    std::string* str_ptr = new std::string("value");

    value* val = new value(str_ptr, "std::string");
    std::map<std::string, value const*> map;
    map.insert(std::pair<std::string, value const*>("flag", val));
    const values vals(map);

    CHECK_EQUAL("std::string", vals.get_value_class_name("flag"));
    delete str_ptr;
  }

  TEST(TestCast)
  {
    const value* value_int = new value("42");
    const value* value_float = new value("4.2");
    const value* value_bool = new value("1");

    std::map<std::string, value const*> map;
    map.insert(std::pair<std::string, value const*>("int", value_int));
    map.insert(std::pair<std::string, value const*>("float", value_float));
    map.insert(std::pair<std::string, value const*>("bool", value_bool));
    const values vals(map);

    CHECK_EQUAL(42, vals.cast<int>("int"));
    CHECK_CLOSE(4.2f, vals.cast<float>("float"), 0.00001);
    CHECK(vals.cast<bool>("bool"));
  }

  TEST(TestFailNoSuchOption)
  {
    std::string* str_ptr = new std::string("value");
    value* val_ptr = new value(str_ptr, "std::string");
    value* val_str = new value("string");
    value* val_int = new value("42");
    value* val_bool = new value("TRUE");

    std::map<std::string, value const*> map;
    map.insert(std::pair<std::string, value const*>("ptr", val_ptr));
    map.insert(std::pair<std::string, value const*>("str", val_str));
    map.insert(std::pair<std::string, value const*>("int", val_int));
    map.insert(std::pair<std::string, value const*>("bool", val_bool));
    const values vals(map);

    CHECK_THROW(vals.get<std::string>("ptrs"), no_such_option_error);
    CHECK_THROW(vals.get_value_class_name("ptrs"), no_such_option_error);
    CHECK_THROW(vals.get("strs"), no_such_option_error);
    CHECK_THROW(vals.cast<int>("in"), no_such_option_error);
    CHECK_THROW(vals.get_boolean("boo"), no_such_option_error);
    delete str_ptr;
  }

  TEST(TestFailPointerToString)
  {
    std::string* str_ptr = new std::string("42");
    value* val = new value(str_ptr, "std::string");
    std::map<std::string, value const*> map;
    map.insert(std::pair<std::string, value const*>("flag", val));
    const values vals(map);

    CHECK_THROW(vals.get("flag"), invalid_type_error);
    CHECK_THROW(vals.cast<int>("flag"), invalid_type_error);
    delete str_ptr;
  }

  TEST(TestFailPointerToDifferentPointer)
  {
    std::string* str_ptr = new std::string("42");
    value* val = new value(str_ptr, "std::string");
    std::map<std::string, value const*> map;
    map.insert(std::pair<std::string, value const*>("flag", val));
    const values vals(map);

    CHECK_THROW(vals.get<int>("flag"), invalid_type_error);
    delete str_ptr;
  }

  TEST(TestFailPointerToBoolean)
  {
    std::string* str_ptr = new std::string("false");
    value* val = new value(str_ptr, "std::string");
    std::map<std::string, value const*> map;
    map.insert(std::pair<std::string, value const*>("flag", val));
    const values vals(map);

    CHECK_THROW(vals.get_boolean("flag"), invalid_type_error);
    delete str_ptr;
  }

  TEST(TestFailStringToBoolean)
  {
    value* val = new value("really true");
    value* val2 = new value("");
    std::map<std::string, value const*> map;
    map.insert(std::pair<std::string, value const*>("flag", val));
    map.insert(std::pair<std::string, value const*>("flag2", val2));
    const values vals(map);

    CHECK_THROW(vals.get_boolean("flag"), invalid_type_error);
    CHECK_THROW(vals.get_boolean("flag2"), invalid_type_error);
  }

  TEST(TestFailStringToPointer)
  {
    value* val = new value("value");
    std::map<std::string, value const*> map;
    map.insert(std::pair<std::string, value const*>("flag", val));
    const values vals(map);

    CHECK_THROW(vals.get<std::string>("flag"), invalid_type_error);
  }

  TEST(TestFailGetValuesClassName)
  {
    value* val = new value("value");
    std::map<std::string, value const*> map;
    map.insert(std::pair<std::string, value const*>("flag", val));
    const values vals(map);

    CHECK_THROW(vals.get_value_class_name("flag"), invalid_type_error);
  }

  TEST(TestFailCast)
  {
    const int* int_ptr = new int(42);
    const value* value_ptr = new value(int_ptr, "int");
    const value* value_int = new value("42");
    const value* value_float = new value("4.2");
    const value* value_string = new value("E42");

    std::map<std::string, value const*> map;
    map.insert(std::pair<std::string, value const*>("ptr", value_ptr));
    map.insert(std::pair<std::string, value const*>("int", value_int));
    map.insert(std::pair<std::string, value const*>("float", value_float));
    map.insert(std::pair<std::string, value const*>("string", value_string));
    const values vals(map);

    CHECK_THROW(vals.cast<int>("ptr"), invalid_type_error);
    CHECK_THROW(vals.cast<bool>("int"), invalid_type_error);
    CHECK_THROW(vals.cast<int>("float"), invalid_type_error);
    CHECK_THROW(vals.cast<float>("string"), invalid_type_error);
    delete int_ptr;
  }
}

