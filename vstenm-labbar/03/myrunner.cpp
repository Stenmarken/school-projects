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
#include "mytest.h"

static MyTestSuite suite_MyTestSuite;

static CxxTest::List Tests_MyTestSuite = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_MyTestSuite( "mytest.h", 9, "MyTestSuite", suite_MyTestSuite, Tests_MyTestSuite );

static class TestDescription_suite_MyTestSuite_test_null : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_null() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 13, "test_null" ) {}
 void runTest() { suite_MyTestSuite.test_null(); }
} testDescription_suite_MyTestSuite_test_null;

static class TestDescription_suite_MyTestSuite_test_assignment_operator_real : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_assignment_operator_real() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 19, "test_assignment_operator_real" ) {}
 void runTest() { suite_MyTestSuite.test_assignment_operator_real(); }
} testDescription_suite_MyTestSuite_test_assignment_operator_real;

static class TestDescription_suite_MyTestSuite_test_assignment_operator_imag : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_assignment_operator_imag() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 25, "test_assignment_operator_imag" ) {}
 void runTest() { suite_MyTestSuite.test_assignment_operator_imag(); }
} testDescription_suite_MyTestSuite_test_assignment_operator_imag;

static class TestDescription_suite_MyTestSuite_test_assignment_operator_complex : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_assignment_operator_complex() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 31, "test_assignment_operator_complex" ) {}
 void runTest() { suite_MyTestSuite.test_assignment_operator_complex(); }
} testDescription_suite_MyTestSuite_test_assignment_operator_complex;

static class TestDescription_suite_MyTestSuite_test_real : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_real() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 37, "test_real" ) {}
 void runTest() { suite_MyTestSuite.test_real(); }
} testDescription_suite_MyTestSuite_test_real;

static class TestDescription_suite_MyTestSuite_test_imag : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_imag() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 43, "test_imag" ) {}
 void runTest() { suite_MyTestSuite.test_imag(); }
} testDescription_suite_MyTestSuite_test_imag;

static class TestDescription_suite_MyTestSuite_test_constructor : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_constructor() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 49, "test_constructor" ) {}
 void runTest() { suite_MyTestSuite.test_constructor(); }
} testDescription_suite_MyTestSuite_test_constructor;

static class TestDescription_suite_MyTestSuite_test_copy_constructor : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_copy_constructor() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 55, "test_copy_constructor" ) {}
 void runTest() { suite_MyTestSuite.test_copy_constructor(); }
} testDescription_suite_MyTestSuite_test_copy_constructor;

static class TestDescription_suite_MyTestSuite_test_abs_both_positive : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_abs_both_positive() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 62, "test_abs_both_positive" ) {}
 void runTest() { suite_MyTestSuite.test_abs_both_positive(); }
} testDescription_suite_MyTestSuite_test_abs_both_positive;

static class TestDescription_suite_MyTestSuite_test_abs_both_negative : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_abs_both_negative() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 67, "test_abs_both_negative" ) {}
 void runTest() { suite_MyTestSuite.test_abs_both_negative(); }
} testDescription_suite_MyTestSuite_test_abs_both_negative;

static class TestDescription_suite_MyTestSuite_test_abs_real_negative : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_abs_real_negative() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 72, "test_abs_real_negative" ) {}
 void runTest() { suite_MyTestSuite.test_abs_real_negative(); }
} testDescription_suite_MyTestSuite_test_abs_real_negative;

static class TestDescription_suite_MyTestSuite_test_abs_imag_negative : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_abs_imag_negative() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 77, "test_abs_imag_negative" ) {}
 void runTest() { suite_MyTestSuite.test_abs_imag_negative(); }
} testDescription_suite_MyTestSuite_test_abs_imag_negative;

static class TestDescription_suite_MyTestSuite_test_plus_equals_both_complex : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_plus_equals_both_complex() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 82, "test_plus_equals_both_complex" ) {}
 void runTest() { suite_MyTestSuite.test_plus_equals_both_complex(); }
} testDescription_suite_MyTestSuite_test_plus_equals_both_complex;

static class TestDescription_suite_MyTestSuite_test_plus_equals_real : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_plus_equals_real() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 90, "test_plus_equals_real" ) {}
 void runTest() { suite_MyTestSuite.test_plus_equals_real(); }
} testDescription_suite_MyTestSuite_test_plus_equals_real;

static class TestDescription_suite_MyTestSuite_test_plus_equals_imag : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_plus_equals_imag() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 97, "test_plus_equals_imag" ) {}
 void runTest() { suite_MyTestSuite.test_plus_equals_imag(); }
} testDescription_suite_MyTestSuite_test_plus_equals_imag;

static class TestDescription_suite_MyTestSuite_test_minus_equals_complex : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_minus_equals_complex() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 104, "test_minus_equals_complex" ) {}
 void runTest() { suite_MyTestSuite.test_minus_equals_complex(); }
} testDescription_suite_MyTestSuite_test_minus_equals_complex;

static class TestDescription_suite_MyTestSuite_test_minus_equals_real : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_minus_equals_real() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 112, "test_minus_equals_real" ) {}
 void runTest() { suite_MyTestSuite.test_minus_equals_real(); }
} testDescription_suite_MyTestSuite_test_minus_equals_real;

static class TestDescription_suite_MyTestSuite_test_minus_equals_imag : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_minus_equals_imag() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 119, "test_minus_equals_imag" ) {}
 void runTest() { suite_MyTestSuite.test_minus_equals_imag(); }
} testDescription_suite_MyTestSuite_test_minus_equals_imag;

static class TestDescription_suite_MyTestSuite_test_times_equals : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_times_equals() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 126, "test_times_equals" ) {}
 void runTest() { suite_MyTestSuite.test_times_equals(); }
} testDescription_suite_MyTestSuite_test_times_equals;

static class TestDescription_suite_MyTestSuite_test_divide_equals_easy : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_divide_equals_easy() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 134, "test_divide_equals_easy" ) {}
 void runTest() { suite_MyTestSuite.test_divide_equals_easy(); }
} testDescription_suite_MyTestSuite_test_divide_equals_easy;

static class TestDescription_suite_MyTestSuite_test_divide_equals_harder : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_divide_equals_harder() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 141, "test_divide_equals_harder" ) {}
 void runTest() { suite_MyTestSuite.test_divide_equals_harder(); }
} testDescription_suite_MyTestSuite_test_divide_equals_harder;

static class TestDescription_suite_MyTestSuite_test_unary_plus : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_unary_plus() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 149, "test_unary_plus" ) {}
 void runTest() { suite_MyTestSuite.test_unary_plus(); }
} testDescription_suite_MyTestSuite_test_unary_plus;

static class TestDescription_suite_MyTestSuite_test_unary_minus : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_unary_minus() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 156, "test_unary_minus" ) {}
 void runTest() { suite_MyTestSuite.test_unary_minus(); }
} testDescription_suite_MyTestSuite_test_unary_minus;

static class TestDescription_suite_MyTestSuite_test_plus_complex : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_plus_complex() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 163, "test_plus_complex" ) {}
 void runTest() { suite_MyTestSuite.test_plus_complex(); }
} testDescription_suite_MyTestSuite_test_plus_complex;

static class TestDescription_suite_MyTestSuite_test_plus_complex_real_left : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_plus_complex_real_left() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 171, "test_plus_complex_real_left" ) {}
 void runTest() { suite_MyTestSuite.test_plus_complex_real_left(); }
} testDescription_suite_MyTestSuite_test_plus_complex_real_left;

static class TestDescription_suite_MyTestSuite_test_plus_complex_real_right : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_plus_complex_real_right() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 178, "test_plus_complex_real_right" ) {}
 void runTest() { suite_MyTestSuite.test_plus_complex_real_right(); }
} testDescription_suite_MyTestSuite_test_plus_complex_real_right;

static class TestDescription_suite_MyTestSuite_test_minus_complex : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_minus_complex() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 185, "test_minus_complex" ) {}
 void runTest() { suite_MyTestSuite.test_minus_complex(); }
} testDescription_suite_MyTestSuite_test_minus_complex;

static class TestDescription_suite_MyTestSuite_test_minus_complex_real_left : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_minus_complex_real_left() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 193, "test_minus_complex_real_left" ) {}
 void runTest() { suite_MyTestSuite.test_minus_complex_real_left(); }
} testDescription_suite_MyTestSuite_test_minus_complex_real_left;

static class TestDescription_suite_MyTestSuite_test_minus_complex_real_right : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_minus_complex_real_right() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 200, "test_minus_complex_real_right" ) {}
 void runTest() { suite_MyTestSuite.test_minus_complex_real_right(); }
} testDescription_suite_MyTestSuite_test_minus_complex_real_right;

static class TestDescription_suite_MyTestSuite_test_multiply_complex : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_multiply_complex() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 207, "test_multiply_complex" ) {}
 void runTest() { suite_MyTestSuite.test_multiply_complex(); }
} testDescription_suite_MyTestSuite_test_multiply_complex;

static class TestDescription_suite_MyTestSuite_test_multiply_complex_real_left : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_multiply_complex_real_left() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 215, "test_multiply_complex_real_left" ) {}
 void runTest() { suite_MyTestSuite.test_multiply_complex_real_left(); }
} testDescription_suite_MyTestSuite_test_multiply_complex_real_left;

static class TestDescription_suite_MyTestSuite_test_multiply_complex_real_right : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_multiply_complex_real_right() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 222, "test_multiply_complex_real_right" ) {}
 void runTest() { suite_MyTestSuite.test_multiply_complex_real_right(); }
} testDescription_suite_MyTestSuite_test_multiply_complex_real_right;

static class TestDescription_suite_MyTestSuite_test_divide_complex : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_divide_complex() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 229, "test_divide_complex" ) {}
 void runTest() { suite_MyTestSuite.test_divide_complex(); }
} testDescription_suite_MyTestSuite_test_divide_complex;

static class TestDescription_suite_MyTestSuite_test_divide_complex_real_left : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_divide_complex_real_left() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 237, "test_divide_complex_real_left" ) {}
 void runTest() { suite_MyTestSuite.test_divide_complex_real_left(); }
} testDescription_suite_MyTestSuite_test_divide_complex_real_left;

static class TestDescription_suite_MyTestSuite_test_equality_complex : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_equality_complex() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 244, "test_equality_complex" ) {}
 void runTest() { suite_MyTestSuite.test_equality_complex(); }
} testDescription_suite_MyTestSuite_test_equality_complex;

static class TestDescription_suite_MyTestSuite_test_equality_real : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_equality_real() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 251, "test_equality_real" ) {}
 void runTest() { suite_MyTestSuite.test_equality_real(); }
} testDescription_suite_MyTestSuite_test_equality_real;

static class TestDescription_suite_MyTestSuite_test_equality_imag : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_equality_imag() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 258, "test_equality_imag" ) {}
 void runTest() { suite_MyTestSuite.test_equality_imag(); }
} testDescription_suite_MyTestSuite_test_equality_imag;

static class TestDescription_suite_MyTestSuite_test_inequality_complex : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_inequality_complex() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 265, "test_inequality_complex" ) {}
 void runTest() { suite_MyTestSuite.test_inequality_complex(); }
} testDescription_suite_MyTestSuite_test_inequality_complex;

static class TestDescription_suite_MyTestSuite_test_inequality_real : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_inequality_real() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 272, "test_inequality_real" ) {}
 void runTest() { suite_MyTestSuite.test_inequality_real(); }
} testDescription_suite_MyTestSuite_test_inequality_real;

static class TestDescription_suite_MyTestSuite_test_inequality_imag : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_inequality_imag() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 279, "test_inequality_imag" ) {}
 void runTest() { suite_MyTestSuite.test_inequality_imag(); }
} testDescription_suite_MyTestSuite_test_inequality_imag;

static class TestDescription_suite_MyTestSuite_test_i_operator_int : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_i_operator_int() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 286, "test_i_operator_int" ) {}
 void runTest() { suite_MyTestSuite.test_i_operator_int(); }
} testDescription_suite_MyTestSuite_test_i_operator_int;

static class TestDescription_suite_MyTestSuite_test_i_operator_double : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_i_operator_double() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 293, "test_i_operator_double" ) {}
 void runTest() { suite_MyTestSuite.test_i_operator_double(); }
} testDescription_suite_MyTestSuite_test_i_operator_double;

static class TestDescription_suite_MyTestSuite_test_left_shift : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_left_shift() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 299, "test_left_shift" ) {}
 void runTest() { suite_MyTestSuite.test_left_shift(); }
} testDescription_suite_MyTestSuite_test_left_shift;

static class TestDescription_suite_MyTestSuite_test_complicated_left_shift : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_complicated_left_shift() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 307, "test_complicated_left_shift" ) {}
 void runTest() { suite_MyTestSuite.test_complicated_left_shift(); }
} testDescription_suite_MyTestSuite_test_complicated_left_shift;

static class TestDescription_suite_MyTestSuite_test_right_shift_complex : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_right_shift_complex() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 314, "test_right_shift_complex" ) {}
 void runTest() { suite_MyTestSuite.test_right_shift_complex(); }
} testDescription_suite_MyTestSuite_test_right_shift_complex;

static class TestDescription_suite_MyTestSuite_test_right_shift_real_easy : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_right_shift_real_easy() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 322, "test_right_shift_real_easy" ) {}
 void runTest() { suite_MyTestSuite.test_right_shift_real_easy(); }
} testDescription_suite_MyTestSuite_test_right_shift_real_easy;

static class TestDescription_suite_MyTestSuite_test_right_shift_real_harder : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_right_shift_real_harder() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 330, "test_right_shift_real_harder" ) {}
 void runTest() { suite_MyTestSuite.test_right_shift_real_harder(); }
} testDescription_suite_MyTestSuite_test_right_shift_real_harder;

static class TestDescription_suite_MyTestSuite_test_chain_operations : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_chain_operations() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 338, "test_chain_operations" ) {}
 void runTest() { suite_MyTestSuite.test_chain_operations(); }
} testDescription_suite_MyTestSuite_test_chain_operations;

#include <cxxtest/Root.cpp>
const char* CxxTest::RealWorldDescription::_worldName = "cxxtest";
