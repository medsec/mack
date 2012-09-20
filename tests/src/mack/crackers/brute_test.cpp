#include <unittest++/UnitTest++.h>
#include <string>
#include <stdlib.h>
#include <mack/core/cracker.hpp>
#include <mack/options/types/crackers.hpp>
#include <mack/options/options.hpp>
#include <mack/options/parser.hpp>
#include <mack/options/values.hpp>
#include <mack/options/exceptions.hpp>
#include <iterator>
#include <mack/callbacks/container_callback.h>
#include <mack/targetloaders/default_loader.hpp>
#include <fstream>

SUITE(BruteCracker_MD5){
	TEST(CheckIfOptionsMissing)
	{
		std::cout << "BruteCracker_MD5::CheckIfOptionsMissing\n" << std::endl;

    mack::options::types::crackers crackers;
    mack::options::options* options = crackers.get_options("mack::Brute_Cracker");
		CHECK_THROW(options->create(), mack::options::no_value_error);
	}

	TEST(FindAllPreimages_3)
	{
		std::cout << "BruteCracker_MD5::FindTestPreimages_3\n" << std::endl;
		//Set Command Line Parameters
		//./mack -C Brute_Cracker --cracker.length 3 --cracker.keytable ../../src/keys/char_tables/visible_ascii.txt -a md5 -t targets/md5_3_hash
		int Xargc = 12;
		char* Xargv[Xargc];
		Xargv[0] = (char*)"mack";
		Xargv[1] = (char*)"-C";
		Xargv[2] = (char*)"Brute_Cracker";
		Xargv[3] = (char*)"--cracker.length";
		Xargv[4] = (char*)"3";
		Xargv[5] = (char*)"--cracker.keytable";
		Xargv[6] = (char*)"resources/char_tables/visible_ascii.txt";
		Xargv[7] = (char*)"-a";
		Xargv[8] = (char*)"md5_algorithm";
		Xargv[9] = (char*)"--cracker.target-file";
		Xargv[10] = (char*)"tests/resources/targets/md5_3_hash";
		Xargv[11] = (char*)"-o=container_callback";

		mack::options::parser parser(Xargc, Xargv, "mack_the_knife");
		mack::options::values const* values = parser.parse();

		mack::core::Cracker* cracker = values->get<mack::core::Cracker>("cracker");

		//found preimages are stored in a map via the container callback
		mack::callbacks::container_callback* callback =
		  (mack::callbacks::container_callback*)
			values->get<mack::callbacks::Callback>("output");

		mack::targetloaders::Target_Loader* target_loader=
			(mack::targetloaders::Default_Loader*)
			values->get<mack::targetloaders::Target_Loader>("loader");

		cracker->crack(callback, target_loader);

		std::map<std::string,mack::core::candidate> found_map = callback->get_container();
		std::map<std::string,mack::core::candidate>::iterator it;

		//open preimage file and compare to found preimages
		const char* file_path_hash = "targets/md5_3_hash";
		const char* file_path_pre = "targets/md5_3_pre";
		std::fstream fileh(file_path_hash);
		std::fstream filep(file_path_pre);

		std::string lineh,linep;
		int all_found_hash = 1;
		int all_found_pre = 1;
		while(fileh.good() && filep.good()){
			getline(fileh,lineh);
			getline(filep,linep);
			if(!fileh.eof() && !filep.eof()){
				it = found_map.find(lineh);
				all_found_hash &= (it != found_map.end());
				all_found_pre &= !(linep.compare((char*)(it->second.value)));
			}
		}
		fileh.close();
		filep.close();

		//check if all hashes were hashed
		CHECK(all_found_hash);
		//check if the correct preimages were found
		CHECK(all_found_pre);

		delete cracker;
		delete callback;
		delete target_loader;
	}

	TEST(FindAllPreimagesLowerCase_3)
	{
		std::cout << "BruteCracker_MD5::FindAllPreimagesLowerCase_3\n" << std::endl;
		std::cout << "This could take a while...\n" << std::endl;
		int Xargc = 12;
		char* Xargv[Xargc];
		Xargv[0] = (char*)"mack";
		Xargv[1] = (char*)"-C";
		Xargv[2] = (char*)"Brute_Cracker";
		Xargv[3] = (char*)"--cracker.length";
		Xargv[4] = (char*)"3";
		Xargv[5] = (char*)"--cracker.keytable";
		Xargv[6] = (char*)"resources/char_tables/lower_case.txt";
		Xargv[7] = (char*)"-a";
		Xargv[8] = (char*)"md5_algorithm";
		Xargv[9] = (char*)"--cracker.target-file";
		Xargv[10] = (char*)"tests/resources/targets/md5_lower_3_hash";
		Xargv[11] = (char*)"-o=container_callback";

		mack::options::parser parser(Xargc, Xargv, "mack_the_knife");
    mack::options::values const* values = parser.parse();

		mack::core::Cracker* cracker = values->get<mack::core::Cracker>("cracker");

		//found preimages are stored in a map via the container callback
		mack::callbacks::container_callback* callback =
		  (mack::callbacks::container_callback*)
			values->get<mack::callbacks::Callback>("output");
		mack::targetloaders::Target_Loader* target_loader=
			(mack::targetloaders::Default_Loader*)
			values->get<mack::targetloaders::Target_Loader>("loader");
		cracker->crack(callback, target_loader);

		std::map<std::string,mack::core::candidate> found_map = callback->get_container();
		std::map<std::string,mack::core::candidate>::iterator it;

		//open preimage file and compare to found preimages
		const char* file_path_hash = "targets/md5_lower_3_hash";
		const char* file_path_pre = "targets/md5_lower_3_pre";
		std::fstream fileh(file_path_hash);
		std::fstream filep(file_path_pre);

		std::string lineh,linep;
		int all_found_hash = 1;
		int all_found_pre = 1;
		while(fileh.good() && filep.good()){
			getline(fileh,lineh);
			getline(filep,linep);
			if(!fileh.eof() && !filep.eof()){
				it = found_map.find(lineh);
				all_found_hash &= (it != found_map.end());
				all_found_pre &= !(linep.compare((char*)(it->second.value)));
			}
		}
		fileh.close();
		filep.close();

		//check if all hashes were hashed
		CHECK(all_found_hash);
		//check if the correct preimages were found
		CHECK(all_found_pre);

		delete cracker;
		delete callback;
		delete target_loader;
	}

//	TEST(FindAllPreimagesVisibleAscii_3)
//	{
//		std::cout << "BruteCracker_MD5::FindAllPreimagesVisibleAscii_3\n" << std::endl;
//		std::cout << "This could take a while...\n" << std::endl;
//		int Xargc = 12;
//		char* Xargv[Xargc];
//		Xargv[0] = (char*)"mack";
//		Xargv[1] = (char*)"-C";
//		Xargv[2] = (char*)"Brute_Cracker";
//		Xargv[3] = (char*)"--cracker.length";
//		Xargv[4] = (char*)"3";
//		Xargv[5] = (char*)"--cracker.keytable";
//		Xargv[6] = (char*)"resources/char_tables/visible_ascii.txt";
//		Xargv[7] = (char*)"-a";
//		Xargv[8] = (char*)"md5_algorithm";
//		Xargv[9] = (char*)"--cracker.target-file";
//		Xargv[10] = (char*)"tests/resources/targets/md5_visible_3_hash";
//		Xargv[11] = (char*)"-o=container_callback";
//
//		mack::options::parser parser(Xargc, Xargv, "mack_the_knife");
//	mack::options::values const* values = parser.parse();
//
//		mack::core::Cracker* cracker = values->get<mack::core::Cracker>("cracker");
//
//		//found preimages are stored in a map via the container callback
//	mack::callbacks::container_callback* callback =
//	  (mack::callbacks::container_callback*)
//		values->get<mack::callbacks::Callback>("output");
//		mack::targetloaders::Target_Loader* target_loader=
//			(mack::targetloaders::Default_Loader*)
//			values->get<mack::targetloaders::Target_Loader>("loader");
//		cracker->crack(callback, target_loader);
//
//		std::map<std::string,mack::core::candidate> found_map = callback->get_container();
//		std::map<std::string,mack::core::candidate>::iterator it;
//
//		//open preimage file and compare to found preimages
//		const char* file_path_hash = "targets/md5_visible_3_hash";
//		const char* file_path_pre = "targets/md5_visible_3_pre";
//		std::fstream fileh(file_path_hash);
//		std::fstream filep(file_path_pre);
//
//		std::string lineh,linep;
//		int all_found_hash = 1;
//		int all_found_pre = 1;
//		while(fileh.good() && filep.good()){
//			getline(fileh,lineh);
//			getline(filep,linep);
//			if(!fileh.eof() && !filep.eof()){
//				it = found_map.find(lineh);
//				all_found_hash &= (it != found_map.end());
//				all_found_pre &= !(linep.compare((char*)(it->second.value)));
//			}
//		}
//		fileh.close();
//		filep.close();
//
//		//check if all hashes were hashed
//		CHECK(all_found_hash);
//		//check if the correct preimages were found
//		CHECK(all_found_pre);
//
//		delete cracker;
//		delete callback;
//		delete target_loader;
//	}

	TEST(FindAllPreimagesVisibleAsciiRandom_3)
	{
		std::cout << "BruteCracker_MD5::FindAllPreimagesVisibleAsciiRandom_3\n" << std::endl;
		std::cout << "This could take a while...\n" << std::endl;
		int Xargc = 12;
		char* Xargv[Xargc];
		Xargv[0] = (char*)"mack";
		Xargv[1] = (char*)"-C";
		Xargv[2] = (char*)"Brute_Cracker";
		Xargv[3] = (char*)"--cracker.length";
		Xargv[4] = (char*)"3";
		Xargv[5] = (char*)"--cracker.keytable";
		Xargv[6] = (char*)"resources/char_tables/visible_ascii.txt";
		Xargv[7] = (char*)"-a";
		Xargv[8] = (char*)"md5_algorithm";
		Xargv[9] = (char*)"--cracker.target-file";
		Xargv[10] = (char*)"tests/resources/targets/md5_visible_rand_3_hash";
		Xargv[11] = (char*)"-o=container_callback";

		mack::options::parser parser(Xargc, Xargv, "mack_the_knife");
	mack::options::values const* values = parser.parse();

		mack::core::Cracker* cracker = values->get<mack::core::Cracker>("cracker");

		//found preimages are stored in a map via the container callback
	mack::callbacks::container_callback* callback =
	  (mack::callbacks::container_callback*)
		values->get<mack::callbacks::Callback>("output");
		mack::targetloaders::Target_Loader* target_loader=
			(mack::targetloaders::Default_Loader*)
			values->get<mack::targetloaders::Target_Loader>("loader");
		cracker->crack(callback, target_loader);

		std::map<std::string,mack::core::candidate> found_map = callback->get_container();
		std::map<std::string,mack::core::candidate>::iterator it;

		//open preimage file and compare to found preimages
		const char* file_path_hash = "targets/md5_visible_rand_3_hash";
		const char* file_path_pre = "targets/md5_visible_rand_3_pre";
		std::fstream fileh(file_path_hash);
		std::fstream filep(file_path_pre);

		std::string lineh,linep;
		int all_found_hash = 1;
		int all_found_pre = 1;
		while(fileh.good() && filep.good()){
			getline(fileh,lineh);
			getline(filep,linep);
			if(!fileh.eof() && !filep.eof()){
				it = found_map.find(lineh);
				all_found_hash &= (it != found_map.end());
				all_found_pre &= !(linep.compare((char*)(it->second.value)));
			}
		}
		fileh.close();
		filep.close();

		//check if all hashes were hashed
		CHECK(all_found_hash);
		//check if the correct preimages were found
		CHECK(all_found_pre);

		delete cracker;
		delete callback;
		delete target_loader;
	}
}

SUITE(BruteCracker_SHA256){


	TEST(FindAllPreimages_3)
	{
		std::cout << "BruteCracker_SHA256::FindTestPreimages_3\n" << std::endl;
		//Set Command Line Parameters
		//./mack -c brute_c --cracker.length 3 --cracker.keytable ../../src/keys/char_tables/visible_ascii.txt -a sha256 -t targets/sha256_3_hash
		int Xargc = 12;
		char* Xargv[Xargc];
		Xargv[0] = (char*)"mack";
		Xargv[1] = (char*)"-C";
		Xargv[2] = (char*)"Brute_Cracker";
		Xargv[3] = (char*)"--cracker.length";
		Xargv[4] = (char*)"3";
		Xargv[5] = (char*)"--cracker.keytable";
		Xargv[6] = (char*)"resources/char_tables/visible_ascii.txt";
		Xargv[7] = (char*)"-a";
		Xargv[8] = (char*)"sha256_algorithm";
		Xargv[9] = (char*)"--cracker.target-file";
		Xargv[10] = (char*)"tests/resources/targets/sha256_3_hash";
		Xargv[11] = (char*)"-o=container_callback";

		mack::options::parser parser(Xargc, Xargv, "mack_the_knife");
    mack::options::values const* values = parser.parse();

		mack::core::Cracker* cracker = values->get<mack::core::Cracker>("cracker");

		//found preimages are stored in a map via the container callback
    mack::callbacks::container_callback* callback =
      (mack::callbacks::container_callback*)
        values->get<mack::callbacks::Callback>("output");
    	mack::targetloaders::Target_Loader* target_loader=
			(mack::targetloaders::Default_Loader*)
			values->get<mack::targetloaders::Target_Loader>("loader");
		cracker->crack(callback, target_loader);

		std::map<std::string,mack::core::candidate> found_map = callback->get_container();
		std::map<std::string,mack::core::candidate>::iterator it;

		//open preimage file and compare to found preimages
		const char* file_path_hash = "targets/sha256_3_hash";
		const char* file_path_pre = "targets/sha256_3_pre";
		std::fstream fileh(file_path_hash);
		std::fstream filep(file_path_pre);

		std::string lineh,linep;
		int all_found_hash = 1;
		int all_found_pre = 1;
		while(fileh.good() && filep.good()){
			getline(fileh,lineh);
			getline(filep,linep);
			if(!fileh.eof() && !filep.eof()){
				it = found_map.find(lineh);
				all_found_hash &= (it != found_map.end());
				all_found_pre &= !(linep.compare((char*)(it->second.value)));
			}
		}
		fileh.close();
		filep.close();

		//check if all hashes were hashed
		CHECK(all_found_hash);
		//check if the correct preimages were found
		CHECK(all_found_pre);

		delete cracker;
		delete callback;
		delete target_loader;
	}

	TEST(FindAllPreimagesLowerCase_3)
	{
		std::cout << "BruteCracker_SHA256::FindAllPreimagesLowerCase_3\n" << std::endl;
		std::cout << "This could take a while...\n" << std::endl;
		//Set Command Line Parameters
		//./mack -c brute_c --cracker.length 3 --cracker.keytable ../../src/keys/char_tables/visible_ascii.txt -a md5 -t targets/md5_1_hash
		int Xargc = 12;
		char* Xargv[Xargc];
		Xargv[0] = (char*)"mack";
		Xargv[1] = (char*)"-C";
		Xargv[2] = (char*)"Brute_Cracker";
		Xargv[3] = (char*)"--cracker.length";
		Xargv[4] = (char*)"3";
		Xargv[5] = (char*)"--cracker.keytable";
		Xargv[6] = (char*)"resources/char_tables/lower_case.txt";
		Xargv[7] = (char*)"-a";
		Xargv[8] = (char*)"sha256_algorithm";
		Xargv[9] = (char*)"--cracker.target-file";
		Xargv[10] = (char*)"tests/resources/targets/sha256_lower_3_hash";
		Xargv[11] = (char*)"-o=container_callback";

		mack::options::parser parser(Xargc, Xargv, "mack_the_knife");
    mack::options::values const* values = parser.parse();

		mack::core::Cracker* cracker = values->get<mack::core::Cracker>("cracker");

		//found preimages are stored in a map via the container callback
    mack::callbacks::container_callback* callback =
      (mack::callbacks::container_callback*)
        values->get<mack::callbacks::Callback>("output");
    	mack::targetloaders::Target_Loader* target_loader=
			(mack::targetloaders::Default_Loader*)
			values->get<mack::targetloaders::Target_Loader>("loader");
		cracker->crack(callback, target_loader);

		std::map<std::string,mack::core::candidate> found_map = callback->get_container();
		std::map<std::string,mack::core::candidate>::iterator it;

		//open preimage file and compare to found preimages
		const char* file_path_hash = "targets/sha256_lower_3_hash";
		const char* file_path_pre = "targets/sha256_lower_3_pre";
		std::fstream fileh(file_path_hash);
		std::fstream filep(file_path_pre);

		std::string lineh,linep;
		int all_found_hash = 1;
		int all_found_pre = 1;
		while(fileh.good() && filep.good()){
			getline(fileh,lineh);
			getline(filep,linep);
			if(!fileh.eof() && !filep.eof()){
				it = found_map.find(lineh);
				all_found_hash &= (it != found_map.end());
				all_found_pre &= !(linep.compare((char*)(it->second.value)));
			}
		}
		fileh.close();
		filep.close();

		//check if all hashes were hashed
		CHECK(all_found_hash);
		//check if the correct preimages were found
		CHECK(all_found_pre);

		delete cracker;
		delete callback;
		delete target_loader;
	}

	TEST(FindAllPreimagesVisibleAscii_3)
	{
		std::cout << "BruteCracker_SHA256::FindAllPreimagesVisibleAscii_3\n" << std::endl;
		std::cout << "This could take a while...\n" << std::endl;
		//Set Command Line Parameters
		//./mack -c brute_c --cracker.length 3 --cracker.keytable ../../src/keys/char_tables/visible_ascii.txt -a md5 -t targets/md5_visible_3_hash
		int Xargc = 12;
		char* Xargv[Xargc];
		Xargv[0] = (char*)"mack";
		Xargv[1] = (char*)"-C";
		Xargv[2] = (char*)"Brute_Cracker";
		Xargv[3] = (char*)"--cracker.length";
		Xargv[4] = (char*)"3";
		Xargv[5] = (char*)"--cracker.keytable";
		Xargv[6] = (char*)"resources/char_tables/visible_ascii.txt";
		Xargv[7] = (char*)"-a";
		Xargv[8] = (char*)"sha256_algorithm";
		Xargv[9] = (char*)"--cracker.target-file";
		Xargv[10] = (char*)"tests/resources/targets/sha256_visible_3_hash";
		Xargv[11] = (char*)"-o=container_callback";

		mack::options::parser parser(Xargc, Xargv, "mack_the_knife");
    mack::options::values const* values = parser.parse();

		mack::core::Cracker* cracker = values->get<mack::core::Cracker>("cracker");

		//found preimages are stored in a map via the container callback
    mack::callbacks::container_callback* callback =
      (mack::callbacks::container_callback*)
        values->get<mack::callbacks::Callback>("output");
    	mack::targetloaders::Target_Loader* target_loader=
			(mack::targetloaders::Default_Loader*)
			values->get<mack::targetloaders::Target_Loader>("loader");
		cracker->crack(callback, target_loader);

		std::map<std::string,mack::core::candidate> found_map = callback->get_container();
		std::map<std::string,mack::core::candidate>::iterator it;

		//open preimage file and compare to found preimages
		const char* file_path_hash = "targets/sha256_visible_3_hash";
		const char* file_path_pre = "targets/sha256_visible_3_pre";
		std::fstream fileh(file_path_hash);
		std::fstream filep(file_path_pre);

		std::string lineh,linep;
		int all_found_hash = 1;
		int all_found_pre = 1;
		while(fileh.good() && filep.good()){
			getline(fileh,lineh);
			getline(filep,linep);
			if(!fileh.eof() && !filep.eof()){
				it = found_map.find(lineh);
				all_found_hash &= (it != found_map.end());
				all_found_pre &= !(linep.compare((char*)(it->second.value)));
			}
		}
		fileh.close();
		filep.close();

		//check if all hashes were hashed
		CHECK(all_found_hash);
		//check if the correct preimages were found
		CHECK(all_found_pre);

		delete cracker;
		delete callback;
		delete target_loader;
	}

	TEST(FindAllPreimagesVisibleAsciiRandom_3)
	{
		std::cout << "BruteCracker_SHA256::FindAllPreimagesVisibleAsciiRandom_3\n" << std::endl;
		std::cout << "This could take a while...\n" << std::endl;
		//Set Command Line Parameters
		//./mack -c brute_c --cracker.length 3 --cracker.keytable ../../src/keys/char_tables/visible_ascii.txt -a md5 -t targets/md5_visible_3_hash
		int Xargc = 12;
		char* Xargv[Xargc];
		Xargv[0] = (char*)"mack";
		Xargv[1] = (char*)"-C";
		Xargv[2] = (char*)"Brute_Cracker";
		Xargv[3] = (char*)"--cracker.length";
		Xargv[4] = (char*)"3";
		Xargv[5] = (char*)"--cracker.keytable";
		Xargv[6] = (char*)"resources/char_tables/visible_ascii.txt";
		Xargv[7] = (char*)"-a";
		Xargv[8] = (char*)"sha256_algorithm";
		Xargv[9] = (char*)"--cracker.target-file";
		Xargv[10] = (char*)"tests/resources/targets/sha256_visible_rand_3_hash";
		Xargv[11] = (char*)"-o=container_callback";

		mack::options::parser parser(Xargc, Xargv, "mack_the_knife");
	mack::options::values const* values = parser.parse();

		mack::core::Cracker* cracker = values->get<mack::core::Cracker>("cracker");

		//found preimages are stored in a map via the container callback
	mack::callbacks::container_callback* callback =
	  (mack::callbacks::container_callback*)
		values->get<mack::callbacks::Callback>("output");
		mack::targetloaders::Target_Loader* target_loader=
			(mack::targetloaders::Default_Loader*)
			values->get<mack::targetloaders::Target_Loader>("loader");
		cracker->crack(callback, target_loader);

		std::map<std::string,mack::core::candidate> found_map = callback->get_container();
		std::map<std::string,mack::core::candidate>::iterator it;

		//open preimage file and compare to found preimages
		const char* file_path_hash = "targets/sha256_visible_rand_3_hash";
		const char* file_path_pre = "targets/sha256_visible_rand_3_pre";
		std::fstream fileh(file_path_hash);
		std::fstream filep(file_path_pre);

		std::string lineh,linep;
		int all_found_hash = 1;
		int all_found_pre = 1;
		while(fileh.good() && filep.good()){
			getline(fileh,lineh);
			getline(filep,linep);
			if(!fileh.eof() && !filep.eof()){
				it = found_map.find(lineh);
				all_found_hash &= (it != found_map.end());
				all_found_pre &= !(linep.compare((char*)(it->second.value)));
			}
		}
		fileh.close();
		filep.close();

		//check if all hashes were hashed
		CHECK(all_found_hash);
		//check if the correct preimages were found
		CHECK(all_found_pre);

		delete cracker;
		delete callback;
		delete target_loader;
	}
}
