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

SUITE(DictionaryCracker){

	TEST(CheckIfOptionsMissing)
	{
		std::cout << "DictionaryCracker::CheckIfOptionsMissing\n" << std::endl;

    mack::options::types::crackers crackers;
    mack::options::options* options = crackers.get_options("mack::Dictionary_Cracker");
		CHECK_THROW(options->create(), mack::options::no_value_error);
	}


	TEST(FindFirstWordEnglish)
	{

		std::cout << "DictionaryCracker::FindFirstWordEnglish\n" << std::endl;
		std::cout << "This could take a while...\n" << std::endl;
		//Set Command Line Parameters
		//./mack -c dict_c -D dict/en.dict -a md5 -t targets/dict_first_hash
		int Xargc = 10;
		char* Xargv[Xargc];
		Xargv[0] = (char*)"mack";
		Xargv[1] = (char*)"-C";
		Xargv[2] = (char*)"Dictionary_Cracker";
		Xargv[3] = (char*)"-D";
		Xargv[4] = (char*)"tests/resources/dict/en.dict";
		Xargv[5] = (char*)"-a";
		Xargv[6] = (char*)"md5_algorithm";
		Xargv[7] = (char*)"--cracker.target-file";
		Xargv[8] = (char*)"tests/resources/targets/dict_first_hash";
		Xargv[9] = (char*)"-o=container_callback";

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
		cracker->crack(callback,target_loader);

		std::map<std::string,mack::core::candidate> found_map = callback->get_container();
		std::map<std::string,mack::core::candidate>::iterator it;

		//open preimage file and compare to found preimages
		const char* file_path_hash = "targets/dict_first_hash";
		const char* file_path_pre = "targets/dict_first_pre";
		std::fstream fileh(file_path_hash);
		std::fstream filep(file_path_pre);

		std::string lineh,linep;
		int all_found_hash = 1;
		int all_found_pre = 1;
		while(fileh.good() && filep.good()){
			getline(fileh,lineh);
			getline(filep,linep);
			it = found_map.find(lineh);
			all_found_hash &= (it != found_map.end());
			all_found_pre &= !(linep.compare((char*)(it->second.value)));
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

		std::cout << "DONE: DictionaryCracker::FindFirstWordEnglish\n" << std::endl;

	}

	TEST(FindLastWordEnglish)
	{

		std::cout << "DictionaryCracker::FindLastWordEnglish\n" << std::endl;
		std::cout << "This could take a while...\n" << std::endl;
		//Set Command Line Parameters
		//./mack -c dict_c -D dict/en.dict -a md5 -t targets/dict_first_hash
		int Xargc = 10;
		char* Xargv[Xargc];
		Xargv[0] = (char*)"mack";
		Xargv[1] = (char*)"-C";
		Xargv[2] = (char*)"Dictionary_Cracker";
		Xargv[3] = (char*)"-D";
		Xargv[4] = (char*)"tests/resources/dict/en.dict";
		Xargv[5] = (char*)"-a";
		Xargv[6] = (char*)"md5_algorithm";
		Xargv[7] = (char*)"--cracker.target-file";
		Xargv[8] = (char*)"tests/resources/targets/dict_last_hash";
		Xargv[9] = (char*)"-o=container_callback";

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
		cracker->crack(callback,target_loader);

		std::map<std::string,mack::core::candidate> found_map = callback->get_container();
		std::map<std::string,mack::core::candidate>::iterator it;

		//open preimage file and compare to found preimages
		const char* file_path_hash = "targets/dict_last_hash";
		const char* file_path_pre = "targets/dict_last_pre";
		std::fstream fileh(file_path_hash);
		std::fstream filep(file_path_pre);

		std::string lineh,linep;
		int all_found_hash = 1;
		int all_found_pre = 1;
		while(fileh.good() && filep.good()){
			getline(fileh,lineh);
			getline(filep,linep);
			it = found_map.find(lineh);
			all_found_hash &= (it != found_map.end());
			all_found_pre &= !(linep.compare((char*)(it->second.value)));
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

		std::cout << "DONE: DictionaryCracker::FindLastWordEnglish\n" << std::endl;

	}

	TEST(FindRandWordEnglish)
	{

		std::cout << "DictionaryCracker::FindRandWordEnglish\n" << std::endl;
		std::cout << "This could take a while...\n" << std::endl;
		//Set Command Line Parameters
		//./mack -c dict_c -D dict/en.dict -a md5 -t targets/dict_rand_hash
		int Xargc = 10;
		char* Xargv[Xargc];
		Xargv[0] = (char*)"mack";
		Xargv[1] = (char*)"-C";
		Xargv[2] = (char*)"Dictionary_Cracker";
		Xargv[3] = (char*)"-D";
		Xargv[4] = (char*)"tests/resources/dict/en.dict";
		Xargv[5] = (char*)"-a";
		Xargv[6] = (char*)"md5_algorithm";
		Xargv[7] = (char*)"--cracker.target-file";
		Xargv[8] = (char*)"tests/resources/targets/dict_rand_hash";
		Xargv[9] = (char*)"-o=container_callback";

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
		cracker->crack(callback,target_loader);

		std::map<std::string,mack::core::candidate> found_map = callback->get_container();
		std::map<std::string,mack::core::candidate>::iterator it;

		//open preimage file and compare to found preimages
		const char* file_path_hash = "targets/dict_rand_hash";
		const char* file_path_pre = "targets/dict_rand_pre";
		std::fstream fileh(file_path_hash);
		std::fstream filep(file_path_pre);

		std::string lineh,linep;
		int all_found_hash = 1;
		int all_found_pre = 1;
		while(fileh.good() && filep.good()){
			getline(fileh,lineh);
			getline(filep,linep);
			it = found_map.find(lineh);
			all_found_hash &= (it != found_map.end());
			all_found_pre &= !(linep.compare((char*)(it->second.value)));
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

		std::cout << "DONE: DictionaryCracker::FindRandWordEnglish\n" << std::endl;

	}

	TEST(FindNumberWordGerman)
	{

		std::cout << "DictionaryCracker::FindNumberWordGerman\n" << std::endl;
		std::cout << "This could take a while...\n" << std::endl;
		//Set Command Line Parameters
		//bin/mack_the_knife -C Dictionary_Cracker -D tests/resources/dict/de.dict -n 2 -a md5_algorithm -t=Default_Loader -cracker.target-file tests/resources/targets/dict_number_hash -o=container_callback
		int Xargc = 12;
		char* Xargv[Xargc];
		Xargv[0] = (char*)"mack";
		Xargv[1] = (char*)"-C";
		Xargv[2] = (char*)"Dictionary_Cracker";
		Xargv[3] = (char*)"-D";
		Xargv[4] = (char*)"tests/resources/dict/de.dict";
		Xargv[5] = (char*)"-n";
		Xargv[6] = (char*)"2";
		Xargv[7] = (char*)"-a";
		Xargv[8] = (char*)"md5_algorithm";
		Xargv[9] = (char*)"--cracker.target-file";
		Xargv[10] = (char*)"tests/resources/targets/dict_number_hash";
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
		cracker->crack(callback,target_loader);

		std::map<std::string,mack::core::candidate> found_map = callback->get_container();
		std::map<std::string,mack::core::candidate>::iterator it;

		//open preimage file and compare to found preimages
		const char* file_path_hash = "targets/dict_number_hash";
		const char* file_path_pre = "targets/dict_number_pre";
		std::fstream fileh(file_path_hash);
		std::fstream filep(file_path_pre);

		std::string lineh,linep;
		int all_found_hash = 1;
		int all_found_pre = 1;
		while(fileh.good() && filep.good()){
			getline(fileh,lineh);
			getline(filep,linep);
			it = found_map.find(lineh);
			all_found_hash &= (it != found_map.end());
			all_found_pre &= !(linep.compare((char*)(it->second.value)));
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

		std::cout << "DONE: DictionaryCracker::FindNumberWordGerman\n" << std::endl;

	}

	TEST(FindNumberVaryCaseWordGerman)
	{

		std::cout << "DictionaryCracker::FindNumberVaryCaseWordGerman\n" << std::endl;
		std::cout << "This could take a while...\n" << std::endl;
		//Set Command Line Parameters
		//./mack -c dict_c -D dict/en.dict -a md5 -t targets/dict_rand_hash
		int Xargc = 13;
		char* Xargv[Xargc];
		Xargv[0] = (char*)"mack";
		Xargv[1] = (char*)"-C";
		Xargv[2] = (char*)"Dictionary_Cracker";
		Xargv[3] = (char*)"-D";
		Xargv[4] = (char*)"tests/resources/dict/de.dict";
		Xargv[5] = (char*)"-n";
		Xargv[6] = (char*)"2";
		Xargv[7] = (char*)"-x";
		Xargv[8] = (char*)"-a";
		Xargv[9] = (char*)"md5_algorithm";
		Xargv[10] = (char*)"--cracker.target-file";
		Xargv[11] = (char*)"tests/resources/targets/dict_number_varycase_hash";
		Xargv[12] = (char*)"-o=container_callback";

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
		cracker->crack(callback,target_loader);

		std::map<std::string,mack::core::candidate> found_map = callback->get_container();
		std::map<std::string,mack::core::candidate>::iterator it;

		//open preimage file and compare to found preimages
		const char* file_path_hash = "targets/dict_number_varycase_hash";
		const char* file_path_pre = "targets/dict_number_varycase_pre";
		std::fstream fileh(file_path_hash);
		std::fstream filep(file_path_pre);

		std::string lineh,linep;
		int all_found_hash = 1;
		int all_found_pre = 1;
		while(fileh.good() && filep.good()){
			getline(fileh,lineh);
			getline(filep,linep);
			it = found_map.find(lineh);
			all_found_hash &= (it != found_map.end());
			all_found_pre &= !(linep.compare((char*)(it->second.value)));
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

		std::cout << "DONE: DictionaryCracker::FindNumberVaryCaseWordGerman\n" << std::endl;

	}
}
