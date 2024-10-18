# Inquiry

Write your answers below and include the questions. Do copy relevant code to your answers so that both questions and answers are in context 

## Answers
## What did you learn from this assignment?
I learned a few things from this assignment. I had forgotten how to work with binary trees so I had to relearn
how they worked and how to implement the different functions. This assignment was also much needed training on pointers and general memory management. I feel that I am getting pretty comfortable with memory management with every assignment which is good.

## What was hardest to implement in this assignment?
Figuring out how to delete a node was quite difficult. It took some time to really understand the different cases of it but once that was done it wasn't too difficult to implement.

## If the signature of insert was changed like below, changing the key of p would not have the desired effect, why is that?
```
void insert(Node * p, int key, double value) {   // reference to pointer removed
  if ( p == nullptr ) {
     p = new Node();
     p -> key = key;   // ???
     // ...
```
It wouldn't have the desired effect since modifying p wihtin insert would only modify the copy of the pointer in the function. This is why we have a reference to the pointer in our insert.

## Are you able to pass a value, such as 17, to a const int & parameter ?
Yes, that's fine. A ``const int &`` means that we have a reference to an int that is const. In this case that is just a reference to the number 17.

## How do you check if two objects are equal if they only have operator< and not operator==?
Say that we want to compare ``a`` and ``b``. We can simply use this expression:
```
!(a < b) && !(b < a)
```
This simply states that if ``a`` is not less than ``b`` and ``b`` is not less than ``a``, then they are equal.

## Does a < b and !(b < a) compare the same thing?
No they don't compare the same thing. As an example, consider the following code:
```
a = 2
b = 2
a < b // false
!(b < a) // true
```
