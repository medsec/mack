#include <unittest++/UnitTest++.h>

#include <mack/core/files.hpp>

#include <string>

const boost::filesystem::path resources =
  boost::filesystem::path("tests") /= "resources";

const boost::filesystem::path empty_file = resources / "empty_file.txt";
const boost::filesystem::path non_empty_file = resources / "short_file.txt";
const boost::filesystem::path not_existing_file = resources / "not_exists.txt";
const boost::filesystem::path not_existing_file_no_parent =
  not_existing_file / "not_exists.txt";
const boost::filesystem::path not_existing_file_parent_is_file =
  empty_file / "not_exists.txt";
const boost::filesystem::path directory = resources / "directory";

const std::string empty_file_content = "";
const std::string non_empty_file_content = "This\nis a short\n\nfile.\n";
const std::string nearly_non_empty_file_content_no_newline =
  "This\nis a short\n\nfile.";
const std::string nearly_non_empty_file_content_no_paragraph =
  "This\nis a short\nfile.\n";

SUITE(mack_core_files)
{
  TEST(TestReadEmptyFile)
  {
    const std::string content = mack::core::files::read_file(empty_file);
    CHECK(content.empty());
  }

  TEST(TestReadNonEmptyFile)
  {
    const std::string content = mack::core::files::read_file(non_empty_file);
    CHECK_EQUAL(non_empty_file_content, content);
  }

  TEST(TestReadFileFailFileNotExists)
  {
    boost::filesystem::remove(not_existing_file);
    CHECK_THROW(
        mack::core::files::read_file(not_existing_file),
        mack::core::files::file_not_exists_error);
  }

  TEST(TestReadFileFailFileNotAFile)
  {
    CHECK_THROW(
        mack::core::files::read_file(directory),
        mack::core::files::not_a_file_error);
  }

  TEST(TestIsContentOf)
  {
    CHECK(mack::core::files::is_content_of(
          empty_file, empty_file_content));
    CHECK(mack::core::files::is_content_of(
          non_empty_file, non_empty_file_content));
    CHECK(!mack::core::files::is_content_of(
          empty_file, non_empty_file_content));
    CHECK(!mack::core::files::is_content_of(
          non_empty_file, empty_file_content));
    CHECK(!mack::core::files::is_content_of(
          non_empty_file, nearly_non_empty_file_content_no_newline));
    CHECK(!mack::core::files::is_content_of(
          non_empty_file, nearly_non_empty_file_content_no_paragraph));
  }

  TEST(TestIsContentOfFailFileNotExists)
  {
    boost::filesystem::remove(not_existing_file);
    CHECK_THROW(
        mack::core::files::is_content_of(
          not_existing_file, non_empty_file_content),
        mack::core::files::file_not_exists_error);
  }

  TEST(TestIsContentOfFailFileNotAFile)
  {
    CHECK_THROW(
        mack::core::files::is_content_of(
          directory, non_empty_file_content),
        mack::core::files::not_a_file_error);
  }

  TEST(TestWriteEmptyFile)
  {
    boost::filesystem::remove(not_existing_file);
    mack::core::files::write_file(empty_file_content, not_existing_file);
    CHECK(mack::core::files::is_content_of(
          not_existing_file, empty_file_content));
    boost::filesystem::remove(not_existing_file);
  }

  TEST(TestWriteNonEmptyFile)
  {
    boost::filesystem::remove(not_existing_file);
    mack::core::files::write_file(non_empty_file_content, not_existing_file);
    CHECK(mack::core::files::is_content_of(
          not_existing_file, non_empty_file_content));
    boost::filesystem::remove(not_existing_file);
  }

  TEST(TestOverwriteFile)
  {
    boost::filesystem::remove(not_existing_file);
    mack::core::files::write_file(empty_file_content, not_existing_file);
    mack::core::files::write_file(non_empty_file_content, not_existing_file);
    CHECK(mack::core::files::is_content_of(
          not_existing_file, non_empty_file_content));
    boost::filesystem::remove(not_existing_file);
  }

  TEST(TestWriteFileFailParentNotExists)
  {
    CHECK_THROW(
        mack::core::files::write_file(
          non_empty_file_content,
          not_existing_file_no_parent),
        mack::core::files::file_not_exists_error);
  }

  TEST(TestWriteFileFailParentIsFile)
  {
    CHECK_THROW(
        mack::core::files::write_file(
          non_empty_file_content,
          not_existing_file_parent_is_file),
        mack::core::files::not_a_directory_error);
  }
}

