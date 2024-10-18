# Inquiry

Write your answers below and include the questions. Do copy relevant code to your answers so that both questions and answers are in context 

## Answers

### What did you learn from this assignment?
Not too much I'm afraid. Most of the knowledge I applied in this lab was knowledge I acquired
during the previous lab. One new thing though was overloading the _i operator, which I had never done before.

### What was the hardest part (or parts) in this assignment?
The trickiest part were the operators for complex numbers. That required a bit of googling to get right.

### Which of your class methods or non-member functions returns a copy?
The copy assignment operator and the copy constructor returns a copy.

### Which of your class methods or non-member functions returns a reference?
No method or non-member function returns a reference in my program. You could have it so the
operator+=, operator-=, operator*= and operator/= returns a reference but since there are just two 
values in a Complex object I just return a new Complex object instead.

### How many tests do you need at minimum to test the abs function?
Four since there are two parts to a complex number, and we need to test all combinations of positive and negative values for those parts.

### Describe and motivate how your implementation of operator= works
My implementation of the operator= looks like this.
```
Complex &Complex::operator=(const Complex &other)
{
    real_part = other.real_part;
    imaginary_part = other.imaginary_part;
    return *this;
}
```
It assigns the real and imaginary parts of the right hand side to the left hand side. Then it just returns the left hand side.

### What constructors/functions are called when writing the statement Complex k = 3 + 5_i;
The operator""_i(unsigned long long int d) function is called for the 5_i expression which in turns calls the Complex(double real, double imaginary) function. Then the Complex operator+(const double c1, const Complex &c2) function is called for the 3 + 5_i expression. The copy constructor Complex::Complex(const Complex &rhs) is then called.

## Describe and motivate how your implementation of operator+= -= /= *= = works with regard to return value
For operator+= there is nothing special going on. The function just adds the real and imaginary parts of the rhs to the lhs and returns the lhs. The same goes for operator-=. For operator*= I simply used the formula
(a + bi) * (c + di) = (ac - bd) + (bc + ad)i. For operator /= I used the formula (a + bi) / (c + di) = (ac + bd) / (c^2 + d^2) + (bc - ad)i / (c^2 + d^2). For operator= I just copy the real and imaginary parts of the rhs to the lhs.

## What is the rule of three in C++. Look it up in a C++ book or on the web.
According to cppreference.com the rule of three states: "If a class requires a user-defined destructor, a user-defined copy constructor, or a user-defined copy assignment operator, it almost certainly requires all three".

## With regard to the rule of three, do you need to explicitly define and implement the copy-constructor in this assignment?
Since the lab insstruction said to have the copy constructor and the copy assignment operator defined I 
have defined them. This means that I should define the destructor but I haven't done that because it would
have to be constexpr in order to be able to use the operator""_i function. But destructors can't be constexpr which is a problem. But not definig a destructor in this case is not a big problem since I never use any dynamic memory.

## The literal i has an underscore before it. Why is that? Look it up in the c++11 draft section 17.6.4.3.5 and explain. Has there been any changes regarding this matter in the new c++17 draft
According to 17.6.4.3.5 of the c+11 draft: "Literal suffix identifiers that do not start with an underscore are reserved for future standardization". In other words, any literal without an underscore before it can be
used by the standard in the future. This is why we have to use the _ suffix. There have been no changes regarding this matter in the new c++17 draft.





