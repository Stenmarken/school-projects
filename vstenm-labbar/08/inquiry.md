# Inquiry

Write your answers below and include the questions. Do copy relevant code to your answers so that both questions and answers are in context 

## Answers

### Are there any benefits using template meta programming to calculate prime numbers in compile time compared to the program that generated c++ code for prime numbers?
Yes, there is a benefit to template meta programming since not only the code but the actual prime numbers will be generated
at compile time. For the code that generates code for prime numbers, I think that the generated prime numbers will be known at runtime.

### Can you calculate and assign an array of prime numbers in compile time using constexpr?
Yes, it is possible to do that. An example of that can be found in test_constexpr.cpp (taken from https://stackoverflow.com/questions/66239693/generating-prime-numbers-at-compile-time)

### Can the compiler evaluate if-statements in compile time using constexpr?
Yes, as is explained in this article https://stackoverflow.com/questions/43434491/difference-between-if-constexpr-vs-if, 
``if constexpr`` is evaluated at compile time.

### What did you learn from this assignment?
I learned about template metaprogramming. It was pretty frustrating at first but got easier as I got used to it. The main problem with it though seems to be its high learning curve and its unreadability.





