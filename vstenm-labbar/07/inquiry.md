# Inquiry

Write your answers below and include the questions. Do copy relevant code to your answers so that both questions and answers are in context 

## Answers

### What did you learn from this assignment?
I learned more about iterators and how they are actually implemented in C++. Usually when I'm coding loops and
stuff I use indices instead of iterators since I never really figured out how iterators worked. But now I see 
the power of iterators. 

### What was the most difficult part of this assignment?
The tree traversal was pretty difficult. Also, figuring out how to implement the iterator methods was tricky 
in the beginning.

### Is there another way of implementing the iterator class instead of inheriting from forward_iterator?
Yes, you can implement everything yourself and then define some properties of your iterator. These properties are:
1. iterator_category - what type of iterator it is. In our case forward_iterator_tag
2. difference_type - type of int that defines the distance between the steps of the iterator. This website (https://www.internalpointers.com/post/writing-custom-iterators-modern-cpp) recommended  std::ptrdiff_t so I went with that.
3. value_type - the type of the thing that the iterator iterates over. In our case Node<S, T>
4. pointer - a pointer to the value_type. In our case Node<S, T>*
5. reference - a reference to the value_type. In our case Node<S, T>&

This info is from the website https://www.internalpointers.com/post/writing-custom-iterators-modern-cpp

### How do you keep track of which nodes you already visited when iterating?
For the operator++ I know that if I move up from a left child to node n, then I have visited all the nodes in the left subtree of n
and none in its right subtree. This also means that I haven't visited n and that it's time to visit n. If I move up from a right child to n I know that I have visited n (since you visit n first and then its right subtree).

For the stack, the pushing and popping of the visited nodes keeps track of the visited nodes. It is based on the same principle 
though. First pop everything in the left tree, then the current node, then the right tree.

### What sort of access rights should you use on the Node class?
The members should probably be set to private. This is because outside classes shouldn't mess with the actual members. 
But I have them set to public now so that I don't have to rewrite a lot of helper methods. The best thing is maybe
to have them private and use friend classes/functions.

### If your test classes needs access rights to private members, how should you manage that?
Either by having getter methods or by using friend classes/functions.

### Is it possible to avoid code duplication using const_cast?
Yeah, if you have two identical methods except that one is const and one isn't you can use const_cast to remove the constness.

The code snippet below shows an example of that. It is taken from https://www.reddit.com/r/cpp_questions/comments/o864kv/idiomatic_way_to_avoid_duplicate_code_for/ and shows an example of this. Obviously, this is not very effective for short
methods like this but it is good for more complicated ones.
```
class C
{
private:
    
    int v;
    
    
public:
    
    int& method()
    {
        return const_cast<int&>(static_cast<const C&>(*this).method());
    }
    
    const int& method() const
    {
        return v;
    }
};
 ```


### This assignment has forced you to write a Node class and functions that operates on the Node. The usual object-oriented solution is to have a Binary tree class with member functions that operate on an internal Node class. What benefits would that solution have compared to the assignment?
I think the main thing is just readability. It's easier to understand what's happening when we have a Binary tree object which consists of many Node objects. The methods are then split into these two classes where they fit best. It makes more intuitive sense to structure the code this way.