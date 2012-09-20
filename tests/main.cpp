#include <unittest++/UnitTest++.h>
#include <unittest++/TestReporterStdout.h>

using namespace UnitTest;
int main( int argc, char** argv )
{
  //test only suite that are given as parameters
  if( argc > 1 )
  {
    const TestList& allTests( Test::GetTestList() );

    TestReporterStdout reporter;
    TestRunner runner( reporter );
    int ret = 0;

    for( int i = 1 ; i < argc ; ++i )
    	ret |= runner.RunTestsIf( allTests, argv[i], True(), 0 );
    return ret;
  }
  else
  {
    return RunAllTests();
  }
}

