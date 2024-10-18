/* Generated file, do not edit */

#ifndef CXXTEST_RUNNING
#define CXXTEST_RUNNING
#endif

#define _CXXTEST_HAVE_STD
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
#include "iterator_tests.h"

static MyTestSuite suite_MyTestSuite;

static CxxTest::List Tests_MyTestSuite = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_MyTestSuite( "iterator_tests.h", 7, "MyTestSuite", suite_MyTestSuite, Tests_MyTestSuite );

static class TestDescription_suite_MyTestSuite_testDereference : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_testDereference() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 10, "testDereference" ) {}
 void runTest() { suite_MyTestSuite.testDereference(); }
} testDescription_suite_MyTestSuite_testDereference;

static class TestDescription_suite_MyTestSuite_testAssignment : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_testAssignment() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 22, "testAssignment" ) {}
 void runTest() { suite_MyTestSuite.testAssignment(); }
} testDescription_suite_MyTestSuite_testAssignment;

static class TestDescription_suite_MyTestSuite_testEqualityInequality : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_testEqualityInequality() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 38, "testEqualityInequality" ) {}
 void runTest() { suite_MyTestSuite.testEqualityInequality(); }
} testDescription_suite_MyTestSuite_testEqualityInequality;

static class TestDescription_suite_MyTestSuite_testIncrementOperators : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_testIncrementOperators() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 54, "testIncrementOperators" ) {}
 void runTest() { suite_MyTestSuite.testIncrementOperators(); }
} testDescription_suite_MyTestSuite_testIncrementOperators;

static class TestDescription_suite_MyTestSuite_testSwapOperator : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_testSwapOperator() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 72, "testSwapOperator" ) {}
 void runTest() { suite_MyTestSuite.testSwapOperator(); }
} testDescription_suite_MyTestSuite_testSwapOperator;

static class TestDescription_suite_MyTestSuite_testInorderIncrements : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_testInorderIncrements() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 98, "testInorderIncrements" ) {}
 void runTest() { suite_MyTestSuite.testInorderIncrements(); }
} testDescription_suite_MyTestSuite_testInorderIncrements;

static class TestDescription_suite_MyTestSuite_testOperatorChaining : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_testOperatorChaining() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 142, "testOperatorChaining" ) {}
 void runTest() { suite_MyTestSuite.testOperatorChaining(); }
} testDescription_suite_MyTestSuite_testOperatorChaining;

static class TestDescription_suite_MyTestSuite_testConstIterator : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_testConstIterator() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 168, "testConstIterator" ) {}
 void runTest() { suite_MyTestSuite.testConstIterator(); }
} testDescription_suite_MyTestSuite_testConstIterator;

#include <cxxtest/Root.cpp>
const char* CxxTest::RealWorldDescription::_worldName = "cxxtest";
