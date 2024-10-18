#include <string>

struct Node {
    int key;
    double data;
    Node * right;
    Node * left;
    Node(int k, double d) : key(k), data(d), right(nullptr), left(nullptr) {}
    Node() {}
};

void insert(Node * & p, int key, double to_be_inserted);  // Note: reference to pointer
void remove(Node * & p, const int & key);
const double & find(Node * p, const int & to_be_found);
double & edit(Node * p, const int & to_be_changed);
void delete_tree(Node * & p);
void delete_tree_helper(Node * & p);

unsigned int max_height(Node * p);
unsigned int min_height(Node * p);
unsigned int size(Node * p);
bool is_balanced(Node * p);
void printBT(const std::string& prefix, const Node* node, bool isLeft);
void printBT(const Node* node);

Node * inorder_predecessor(Node * p);
