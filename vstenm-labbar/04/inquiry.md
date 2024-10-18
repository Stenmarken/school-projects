### What did you learn from this assignment?
I learned quite a bit about shared pointers, virtual functions, virtual classes and about diamond inheritance. This lab and the 
second lab are probably the ones where I've learned the most.

### What was most difficult to do in this assignment?
I struggled for a while with the creation of the chesspieces which resulted in not being able to run the overridden functions
in Rook, Queen, Pawn and so on. For a long time I thought it was my implementation with the virutal function that was faulty but it turned out to be the creation of the chesspieces. That was, in my opinion, the most difficult part of this assignment.

### The code relies on virtual functions. Could the code have been written without virtual functions?
As it is pointed out in this Stack overflow article(https://stackoverflow.com/questions/2391679/why-do-we-need-virtual-functions-in-c) the advantage of virtual functions is the fact that we can call functions that exist in subclasses from an object of the superclass. I.e. if A is a ChessPiece of type Pawn. Running A.validMove() will get us the function defined in the Pawn class. 

If we didn't use virtual functions we couldn't have this functionality and would have to specify a lot of methods for every type 
of ChessPiece. It would be a pain but it would be doable.

### Could ChessPiece have been an abstract class?
Yeah it could. An abstract class is a class that can not be instantiated but can act as a base class. Since we're only interested
in instantiating objects from subclasses to ChessPiece, it could work as an abstract class.

### There was no separate unit test for each function in this assignment, instead you tested several functions at once with different boards. What are the benefits and drawbacks for this kind of testing compared to unit tests?

The benefits of unit testing is that it tests the whole code and exposes bugs that are otherwise hard to find. However, a big drawback is that it requires so many testcases to test every edge case of every function. For an assignment like this, it would be very tedious.

Testing different boards has the benefit that you can test the whole program at once instead of writing one test for one function. It is also easier to add new tests since you just make a new board. The drawback is that there can exist bugs and errors behind the surface that are not visible with this form of testing.

### What is the problem with a diamond inheritance?
One major problem with a diamond inheritance is ambiguity. Let's say we have a superclass A that classes B and C both inherit from. Nothing is problematic so far but if we then make a class D that inherits form B and C we have diamond inheritance and a problem. Let's say that we call a method that is overridden in both B and C but not in D. This causes a problem for the compiler since it doesn't know whether to run the method in B or in C. 

### Did you encounter any problem with protected member variables and, if so, how did you solve them?
Yeah sometimes in ChessBoard I want to know the values of these protected variables. My solution to this problem was to create accessor functions for m_x, m_y and m_is_white. 

### Create a queen object and try to set the unnecessaryInt in ChessPiece. What happens?
Since I have defined the parent classes Rook and Bishop as virtual then nothing happens since this means that only one copy of the base class variables is inherited. If I hadn't defined them as virtual then I would get an ambiguity error. 

### Think about if it would be better if the queen had a bishop and a rook (as members) instead of inherited from them?
Yeah that would probably be a good idea. That would solve the diamond inheritance problem and thus remove the ambiguity errors.



