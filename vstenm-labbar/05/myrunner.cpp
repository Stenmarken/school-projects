/* Generated file, do not edit */

#ifndef CXXTEST_RUNNING
#define CXXTEST_RUNNING
#endif

#define _CXXTEST_HAVE_STD
#define _CXXTEST_HAVE_EH
#include <cxxtest/TestListener.h>
#include <cxxtest/TestTracker.h>
#include <cxxtest/TestRunner.h>
#include <cxxtest/RealDescriptions.h>
#include <cxxtest/TestMain.h>
#include <cxxtest/ErrorPrinter.h>

int main( int argc, char *argv[] ) {
 int status;
    CxxTest::ErrorPrinter tmp;
    CxxTest::RealWorldDescription::_worldName = "cxxtest";
    status = CxxTest::Main< CxxTest::ErrorPrinter >( tmp, argc, argv );
    return status;
}
bool suite_MyTestSuite_init = false;
#include "mytest.h"

static MyTestSuite suite_MyTestSuite;

static CxxTest::List Tests_MyTestSuite = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_MyTestSuite( "mytest.h", 6, "MyTestSuite", suite_MyTestSuite, Tests_MyTestSuite );

static class TestDescription_suite_MyTestSuite_testBiggerTree : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_testBiggerTree() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 9, "testBiggerTree" ) {}
 void runTest() { suite_MyTestSuite.testBiggerTree(); }
} testDescription_suite_MyTestSuite_testBiggerTree;

static class TestDescription_suite_MyTestSuite_testEmptyTree : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_testEmptyTree() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 48, "testEmptyTree" ) {}
 void runTest() { suite_MyTestSuite.testEmptyTree(); }
} testDescription_suite_MyTestSuite_testEmptyTree;

static class TestDescription_suite_MyTestSuite_testLeaf : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_testLeaf() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 55, "testLeaf" ) {}
 void runTest() { suite_MyTestSuite.testLeaf(); }
} testDescription_suite_MyTestSuite_testLeaf;

#include <cxxtest/Root.cpp>
const char* CxxTest::RealWorldDescription::_worldName = "cxxtest";
