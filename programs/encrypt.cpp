#include <mack/options/parser.hpp>
#include <mack/options/values.hpp>
#include <mack/options/exit_requested.hpp>
#include <cstdlib>

#include <string>
#include <mack/core/algorithm.cuh>
#include <mack/logging.hpp>

/**
 * @program{encrypt}
 * @brief Using an algorithm on a message.
 * @type_option{a,algorithm,algorithms}
 * Sets the algorithm to be used.
 * @option{m,message}
 * Sets the message to be encrypted.
 */
int
main(int argc, char** argv)
{
  try
  {
    mack::options::parser parser(argc, argv, "encrypt");
    while (true)
    {
      mack::options::values const* program_values = parser.parse();
      mack::core::Algorithm* algorithm =
        program_values->get<mack::core::Algorithm>("algorithm");
      std::string message =
        program_values->get("message");

      // Check message length
      LOG_STREAM_DEBUG("Got message: " << message);
      LOG_STREAM_DEBUG("Checking message length: " << message.length());
      if (message.length() > MAX_CAND_SIZE)
      {
        LOG_STREAM_ERROR("Maximum message length is " << MAX_CAND_SIZE
            << ", but given message '" << message << "' is of length "
            << message.length());
        throw mack::options::exit_requested(EXIT_FAILURE);
      }

      // Create required structures
	    mack::core::candidate candidate;
      candidate.length = message.length();
      memcpy(candidate.value, message.c_str(), candidate.length);
      LOG_DEBUG("Getting result size:");
      const size_t result_length = algorithm->get_target_size(message.length());
      LOG_STREAM_DEBUG("\t" << result_length);
      unsigned char* result = new unsigned char[result_length];

      // Do the work
      LOG_DEBUG("Running algorithm");
      algorithm->compute_target(candidate, result);

      // Print formatted
      LOG_DEBUG("Printing result");
      for(unsigned int i = 0 ; i < result_length; ++i) {
          printf("%02x", result[i] & 0xff); //02 fills leading zeros
      }
      std::cout << std::endl;

      // Cleanup
      LOG_DEBUG("Cleaning up");
      delete algorithm;
      delete[] result;
    }
  }
  catch (mack::options::exit_requested const& e)
  {
    return e.exit_code;
  }
}

