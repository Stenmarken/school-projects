# Inquiry

Write your answers below and include the questions. Do copy relevant code to your answers so that both questions and answers are in context 

## Answers

## What did you learn in this assignment?
I learned a lot about lambda expressions in C++, characterics of smart pointers and about multithreading in C++. Also a bit about general memory management in C++ which I feel that I am getting quite comfortable with now.

## What is a capture list in a lambda expression?
A capture list in a lambda expression is a list containing the outside variables that should be accessible within the lambda function.

## What does & mean inside the capture list?
Having a capture list of ``[&]`` means that all local variables are passed by reference into the lambda function.

## When could capturing data by reference [&] be useful?
It could be useful if we are modifying the elements of a vector. In that case we are avoiding a lot of copying by returning by reference.

## What does {4} in the code below do?
```
A * p = new A {4} ;
```
It is a form of list initialization where ``data`` in ``A`` is initialized to ``4``.

## Why is it a compile error to call foo with a unique_ptr without a move?
```
foo(pa);
```

A unique_ptr cannot, by definition, be passed by value to a function. But we can either pass it by reference or perform a move.

## Was there ever a memory leak when calling foo2 with a shared pointer?

## What is the use of a weak_ptr?
Weak pointers are often used to break the circular dependencies that can occur with smart poitners. We see an example of this in the lab where we have a circular dependency that causes a memory leak that is solved by the use of a weak pointer.

## How do you create a unique_ptr with a special deleter?
It is done like this
```
auto deleter = [](B * p) {delete [] p;};
unique_ptr<B, decltype(deleter)> pb2(new B[2], deleter);
```

Here we also get an example of ``decltype`` where we don't know the type of the expression ``deleter``.

## What is decltype?
Decltype is used to specify the type of an expression ``a`` by running ``decltype(a)``. The difference between ``auto``and ``decltype`` is that ``auto`` works on types and ``decltype`` works on expressions.
(https://stackoverflow.com/questions/18815221/what-is-decltype-and-how-is-it-used)

## What is std::function?
```std::function``` is an example of type erasure. It takes any type of functions or function pointers, erases the type so that we only have to deal with the ```std::function```. As is pointed out in this article, https://stackoverflow.com/questions/51934866/explanation-of-stdfunction, it can be useful to do this if we want to store a list of objects of different types if we only care that they implement one interface.

## What is [this] in the lambda function sent to the condition_variable's wait-method? Why is [this] needed?
It means that it captures the ``this`` pointer which is sent to the lambda function. It is needed to that the lambda function can access the ``hyenasInside`` and ``gnusInside`` variables.

## Why is the lock a local variable? What happens when the variable goes out of scope?
Since a lock_guard only is locked once and the unlocked on destruction, we have it as a local variable so that the variable is destroyed when it goes out of scope.

## What is the difference between unique_lock and lock_guard
The difference between unique_lock and lock_guard is that a unique_lock can be locked and unlocked while a lock_guard is locked once and the unlocked on destruction.

## Do the integers hyenasInside, gnusInside need be atomic?

## Describe how .wait and .notifyall() works. Is any call blocking, and what does blocking mean?

## Write code that iterates over an unordered_map and prints each key and value
From this article, https://stackoverflow.com/questions/50870951/iterating-over-unordered-map-c, you can do

```
for (auto& [key, value]: B) 
    std::cout << key << " " << value; 
```

## When printing an unordered_map, why is the items not printed in the order they were inserted?
Because internally, an unordered map isn't stored with any order in mind. Instead they are stored into buckets and which buckets an element falls into has to do with its key value.

## In what order would the items be printed if a map was used instead of an unordered_map?
According to the reference, https://en.cppreference.com/w/cpp/container/map, the keys are sorted according to the function compare so it's smallest to biggest for numeric types.

## How did you implement turning on/off trace outputs? Compile time, runtime or both? Elaborate over your decision

## What information did you print out in your trace output from the water cave? Elaborate over your decision

## Do you prefer initializing your thread with a function or function object (functor)



