# Inquiry

Write your answers below and include the questions. Do copy relevant code to your answers so that both questions and answers are in context 

## Answers

### What did you learn from this assignment?
I learned a lot about different features in C++ like memory management, operator overloading, different types of constructors and destructors. 

### What was hardest to implement in this assignment?
The operators, constructors and the move operators were the most difficult to implement. Mainly because
they require you to have a good theoretical understanding before you can implement them.

### How many methods/functions have you implemented?
I've implemented about 33 methods/functions. Most of them were specified in the lab but I also added some
functions for my own benefit. Stuff like a smooth print function and a way of comparing matrices.

### How many methods/functions have you tested?
I've tested in total 31 (I didn't double check so this number could be off by one or two) methods since you had to test those according to the lab instructions. 

### Describe and motivate how your implementation of operator= works

1. **Copy assignment operator:**
I first check that the matrices on both sides of the assignment are not the same. If they are the same I don't do anything other than just return the matrix. However, if they are they same I create a a new 
array with the same size as the matrix on the right side of the assignment. Then I use the std::copy
function to copy the values from the matrix on the right hand side to the new array. Having done that
I delete the old array and set the pointer of the matrix on the left hand side to the new array.
Lastly I set the variables of the matrix on the left hand side to the variables of the matrix on the right hand side and then return the value of the left hand side matrix.
Let's call the left hand side matrix m_1 and the right hand side matrix m_2. If I do m_1 = m_2 then I want m_1 to have the same values as m_2 but a change in m_2 should not affect m_1. 

2. **Move assignment operator:**
First we want to check that we aren't moving an object to itself. If we are we just return the object. 
Otherwise we delete the potential memory we have allocated for m_vec before assigning all the data of the moved from object
to the moved to object.

### When do you need to resize in your code?
Any structural changes to the matrix requires a resizing. This is obviously not ideal but it made it easy to 
test different functions that resize the matrix.

### Why do you need to implement a const-version of the accessor operator?
We have a non-const accessor operator so that non-const objects can both read and write to the matrix.
We have a const accessor operator so that const objects can read but not write to the matrix.

### Is it possible to call std::sort() to sort your matrix using begin() and end()?
Yes, for a matrix m you can just call std::sort(m.begin(), m.end()) and it will sort the matrix.

### What other kind of iterators, besides random access iterators, are there in C++?
There are input iterators, output iterators, forward iterators and bidirectional iterators. 
You can only read from an input iterator, only write to an output iterator, only move forward with a forward iterator and only move forward and backward with a bidirectional iterator. 

