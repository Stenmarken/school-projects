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

static class TestDescription_suite_MyTestSuite_test_empty_matrix : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_empty_matrix() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 17, "test_empty_matrix" ) {}
 void runTest() { suite_MyTestSuite.test_empty_matrix(); }
} testDescription_suite_MyTestSuite_test_empty_matrix;

static class TestDescription_suite_MyTestSuite_test_dimension_matrix : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_dimension_matrix() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 27, "test_dimension_matrix" ) {}
 void runTest() { suite_MyTestSuite.test_dimension_matrix(); }
} testDescription_suite_MyTestSuite_test_dimension_matrix;

static class TestDescription_suite_MyTestSuite_test_matrix_rows_cols : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_matrix_rows_cols() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 39, "test_matrix_rows_cols" ) {}
 void runTest() { suite_MyTestSuite.test_matrix_rows_cols(); }
} testDescription_suite_MyTestSuite_test_matrix_rows_cols;

static class TestDescription_suite_MyTestSuite_test_matrix_initializer_list : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_matrix_initializer_list() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 52, "test_matrix_initializer_list" ) {}
 void runTest() { suite_MyTestSuite.test_matrix_initializer_list(); }
} testDescription_suite_MyTestSuite_test_matrix_initializer_list;

static class TestDescription_suite_MyTestSuite_test_matrix_initializer_list_exception : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_matrix_initializer_list_exception() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 60, "test_matrix_initializer_list_exception" ) {}
 void runTest() { suite_MyTestSuite.test_matrix_initializer_list_exception(); }
} testDescription_suite_MyTestSuite_test_matrix_initializer_list_exception;

static class TestDescription_suite_MyTestSuite_test_matrix_copy_constructor : public CxxTest::RealTestDescription {
public:
<<<<<<< HEAD
 TestDescription_suite_MyTestSuite_test_matrix_copy_constructor() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 64, "test_matrix_copy_constructor" ) {}
=======
 TestDescription_suite_MyTestSuite_test_matrix_copy_constructor() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 66, "test_matrix_copy_constructor" ) {}
>>>>>>> master
 void runTest() { suite_MyTestSuite.test_matrix_copy_constructor(); }
} testDescription_suite_MyTestSuite_test_matrix_copy_constructor;

static class TestDescription_suite_MyTestSuite_test_rows : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_rows() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 74, "test_rows" ) {}
 void runTest() { suite_MyTestSuite.test_rows(); }
} testDescription_suite_MyTestSuite_test_rows;

static class TestDescription_suite_MyTestSuite_test_cols : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_cols() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 79, "test_cols" ) {}
 void runTest() { suite_MyTestSuite.test_cols(); }
} testDescription_suite_MyTestSuite_test_cols;

static class TestDescription_suite_MyTestSuite_test_parenthesis_operator : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_parenthesis_operator() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 85, "test_parenthesis_operator" ) {}
 void runTest() { suite_MyTestSuite.test_parenthesis_operator(); }
} testDescription_suite_MyTestSuite_test_parenthesis_operator;

static class TestDescription_suite_MyTestSuite_test_const_parenthesis_operator : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_const_parenthesis_operator() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 93, "test_const_parenthesis_operator" ) {}
 void runTest() { suite_MyTestSuite.test_const_parenthesis_operator(); }
} testDescription_suite_MyTestSuite_test_const_parenthesis_operator;

static class TestDescription_suite_MyTestSuite_test_insert_row : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_insert_row() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 101, "test_insert_row" ) {}
 void runTest() { suite_MyTestSuite.test_insert_row(); }
} testDescription_suite_MyTestSuite_test_insert_row;

static class TestDescription_suite_MyTestSuite_test_append_row : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_append_row() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 114, "test_append_row" ) {}
 void runTest() { suite_MyTestSuite.test_append_row(); }
} testDescription_suite_MyTestSuite_test_append_row;

static class TestDescription_suite_MyTestSuite_test_remove_row : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_remove_row() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 126, "test_remove_row" ) {}
 void runTest() { suite_MyTestSuite.test_remove_row(); }
} testDescription_suite_MyTestSuite_test_remove_row;

static class TestDescription_suite_MyTestSuite_test_insert_column : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_insert_column() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 139, "test_insert_column" ) {}
 void runTest() { suite_MyTestSuite.test_insert_column(); }
} testDescription_suite_MyTestSuite_test_insert_column;

static class TestDescription_suite_MyTestSuite_test_append_column : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_append_column() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 151, "test_append_column" ) {}
 void runTest() { suite_MyTestSuite.test_append_column(); }
} testDescription_suite_MyTestSuite_test_append_column;

static class TestDescription_suite_MyTestSuite_test_matrix_multiplication : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_matrix_multiplication() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 163, "test_matrix_multiplication" ) {}
 void runTest() { suite_MyTestSuite.test_matrix_multiplication(); }
} testDescription_suite_MyTestSuite_test_matrix_multiplication;

static class TestDescription_suite_MyTestSuite_test_matrix_addition : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_matrix_addition() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 177, "test_matrix_addition" ) {}
 void runTest() { suite_MyTestSuite.test_matrix_addition(); }
} testDescription_suite_MyTestSuite_test_matrix_addition;

static class TestDescription_suite_MyTestSuite_test_matrix_subtraction : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_matrix_subtraction() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 191, "test_matrix_subtraction" ) {}
 void runTest() { suite_MyTestSuite.test_matrix_subtraction(); }
} testDescription_suite_MyTestSuite_test_matrix_subtraction;

static class TestDescription_suite_MyTestSuite_test_matrix_multiply_assignment : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_matrix_multiply_assignment() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 204, "test_matrix_multiply_assignment" ) {}
 void runTest() { suite_MyTestSuite.test_matrix_multiply_assignment(); }
} testDescription_suite_MyTestSuite_test_matrix_multiply_assignment;

static class TestDescription_suite_MyTestSuite_test_reset : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_reset() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 218, "test_reset" ) {}
 void runTest() { suite_MyTestSuite.test_reset(); }
} testDescription_suite_MyTestSuite_test_reset;

static class TestDescription_suite_MyTestSuite_test_identity : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_identity() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 230, "test_identity" ) {}
 void runTest() { suite_MyTestSuite.test_identity(); }
} testDescription_suite_MyTestSuite_test_identity;

static class TestDescription_suite_MyTestSuite_test_move_constructor : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_move_constructor() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 241, "test_move_constructor" ) {}
 void runTest() { suite_MyTestSuite.test_move_constructor(); }
} testDescription_suite_MyTestSuite_test_move_constructor;

static class TestDescription_suite_MyTestSuite_test_move_assignment_operator : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_move_assignment_operator() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 254, "test_move_assignment_operator" ) {}
 void runTest() { suite_MyTestSuite.test_move_assignment_operator(); }
} testDescription_suite_MyTestSuite_test_move_assignment_operator;

static class TestDescription_suite_MyTestSuite_test_plus_equal : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_plus_equal() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 266, "test_plus_equal" ) {}
 void runTest() { suite_MyTestSuite.test_plus_equal(); }
} testDescription_suite_MyTestSuite_test_plus_equal;

static class TestDescription_suite_MyTestSuite_test_left_shift_operator : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_left_shift_operator() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 279, "test_left_shift_operator" ) {}
 void runTest() { suite_MyTestSuite.test_left_shift_operator(); }
} testDescription_suite_MyTestSuite_test_left_shift_operator;

static class TestDescription_suite_MyTestSuite_test_right_shift_operator : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_right_shift_operator() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 286, "test_right_shift_operator" ) {}
 void runTest() { suite_MyTestSuite.test_right_shift_operator(); }
} testDescription_suite_MyTestSuite_test_right_shift_operator;

static class TestDescription_suite_MyTestSuite_test_begin : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_begin() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 294, "test_begin" ) {}
 void runTest() { suite_MyTestSuite.test_begin(); }
} testDescription_suite_MyTestSuite_test_begin;

static class TestDescription_suite_MyTestSuite_test_end : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_end() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 302, "test_end" ) {}
 void runTest() { suite_MyTestSuite.test_end(); }
} testDescription_suite_MyTestSuite_test_end;

static class TestDescription_suite_MyTestSuite_test_sort : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_sort() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 310, "test_sort" ) {}
 void runTest() { suite_MyTestSuite.test_sort(); }
} testDescription_suite_MyTestSuite_test_sort;

static class TestDescription_suite_MyTestSuite_test_copy_assignment_operator : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_copy_assignment_operator() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 332, "test_copy_assignment_operator" ) {}
 void runTest() { suite_MyTestSuite.test_copy_assignment_operator(); }
} testDescription_suite_MyTestSuite_test_copy_assignment_operator;

static class TestDescription_suite_MyTestSuite_test_remove_column : public CxxTest::RealTestDescription {
public:
 TestDescription_suite_MyTestSuite_test_remove_column() : CxxTest::RealTestDescription( Tests_MyTestSuite, suiteDescription_MyTestSuite, 346, "test_remove_column" ) {}
 void runTest() { suite_MyTestSuite.test_remove_column(); }
} testDescription_suite_MyTestSuite_test_remove_column;

#include <cxxtest/Root.cpp>
const char* CxxTest::RealWorldDescription::_worldName = "cxxtest";
