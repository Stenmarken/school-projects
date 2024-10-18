#include <iostream>
#include <vector>
#include<algorithm>
#include <memory>

using namespace std;

void lambda_method()
{
    vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    // Print all odd numbers
    auto odd = [](int x) {if (x % 2 == 1) { cout << x << " "; }};
    std::for_each(v.begin(), v.end(), odd);
    cout << endl;

    // Print all numbers
    auto print_all = [](int x) { cout << x << " "; };

    // Multiply all odd numbers with 2
    auto multiply_odd = [](int &x) {if (x % 2 == 1) { x *= 2;}};
    for_each(v.begin(), v.end(), multiply_odd);
    for_each(v.begin(), v.end(), print_all);
    cout << endl;

    // Add size of vector to all elements
    for_each(v.begin(), v.end(), [&v](int &x) mutable {x += v.size(); });
    for_each(v.begin(), v.end(), print_all);
    cout << endl;
    
    // Add outside number to all elements
    int number_to_add = 100;
    auto outside_add = [&number_to_add](int &x) mutable { x += number_to_add; };
    for_each(v.begin(), v.end(), outside_add);
    for_each(v.begin(), v.end(), print_all);
    cout << endl;
}

struct A {
    int data;
};

void foo(unique_ptr<A> p) {
    cout << p->data << endl;
}

void foo2(shared_ptr<A> p) {
    cout << p->data << endl;
}


void smart_pointers_method()
{
    {
        unique_ptr<A> pa(new A {4} );
        cout << pa -> data << endl;
        
        //foo(pa); Compiler error here!
        foo(move(pa)); // No compiler error!

        shared_ptr<A> sa(new A {5});
        cout << sa -> data << endl;
        foo2(sa);            // No memory leek for this one
        //foo2(move(sa));        // No memory leek for this one either

        weak_ptr<A> wa = sa;
        cout << (wa.lock())->data << endl;
    }
}

struct BhasA;

struct AhasB {
    AhasB(shared_ptr<BhasA> b) : m_b(b)  {
        resource = new int[4];
    };

    shared_ptr<BhasA> m_b;
    int * resource;

    ~AhasB() {delete [] resource;}
    AhasB(const AhasB &) = delete;
    void operator=(const AhasB &) = delete;
};

struct BhasA {
    BhasA() {resource = new int[4];};

    // Changed this from shared_ptr to weak_ptr
    weak_ptr<AhasB> m_a;
    int * resource;

    ~BhasA() {delete [] resource;}
    BhasA(const BhasA &) = delete;
    void operator=(const BhasA &) = delete;
};


void circular_dependencies_method()
{
    shared_ptr<BhasA> bptr(new BhasA);
    shared_ptr<AhasB> aptr(new AhasB(bptr));
    bptr->m_a=aptr;
}

struct B {
    B() { b = new int[4]; }

    int * b;
    ~B() { delete [] b; }
    B(const B &) = delete;
    void operator= (const B & ) = delete;
};


void deleter_method()
{
    // Custom deleter for the unique_ptr
    auto deleter = [](B * p) {delete [] p;};
    unique_ptr<B, decltype(deleter)> pb2(new B[2], deleter);
}

int main()
{
    lambda_method();
    smart_pointers_method();
    circular_dependencies_method();
    deleter_method();
}
