#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <numeric>
#include "bintree.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

using namespace std;

auto rng = std::default_random_engine{};

void eight_hundred(vector<int> &vec, vector<int> &max_heights, vector<int> &min_heights, int tree_size)
{
    Node *p = new Node(vec.at(0), vec.at(0));
    for (int i = 1; i <= tree_size; i++)
    {
        insert(p, vec.at(i), vec.at(i));
    }
    max_heights.push_back(max_height(p));
    min_heights.push_back(min_height(p));
    delete_tree(p);
}

void print_heights(vector<int> &max_heights, vector<int> &min_heights)
{
    std::cout << "Max heights: " << std::endl;
    for (int height : max_heights)
    {
        std::cout << height << std::endl;
    }
    std::cout << "\n\n\n"
              << std::endl;

    std::cout << "Min heights: " << std::endl;
    for (int height : min_heights)
    {
        std::cout << height << std::endl;
    }
}

double calc_average_diff_height(vector<int> &max_heights, vector<int> &min_heights, int num_trees)
{
    double sum = 0;
    for (int i = 0; i < num_trees; i++)
    {
        sum += (max_heights.at(i) - min_heights.at(i));
    }
    sum = sum / num_trees;
    return sum;
}

double calc_max_diff(vector<int> &max_heights, vector<int> &min_heights, int num_trees)
{
    double max = max_heights.at(0) - min_heights.at(0);

    for (int i = 1; i < num_trees; i++)
    {
        double elem = max_heights.at(i) - min_heights.at(i);
        if (elem > max)
            max = elem;
    }
    return max;
}

double calc_min_diff(vector<int> &max_heights, vector<int> &min_heights, int num_trees)
{
    double min = max_heights.at(0) - min_heights.at(0);

    for (int i = 1; i < num_trees; i++)
    {
        double elem = max_heights.at(i) - min_heights.at(i);
        if (elem < min)
            min = elem;
    }
    return min;
}

double calc_average_height(vector<int> &max_heights, vector<int> &min_heights, int num_trees)
{
    double sum = 0;

    for (int i = 0; i < num_trees; i++)
    {
        double avg = (max_heights.at(i) + min_heights.at(i)) / 2;
        sum += avg;
    }
    return sum / num_trees;
}

void standard_test(int tree_size, int num_trees, bool use_next_permutation)
{
    std::vector<int> vec;
    std::vector<int> max_heights;
    std::vector<int> min_heights;

    max_heights.reserve(num_trees);
    min_heights.reserve(num_trees);
    unsigned seed = 000504;

    for (int i = 0; i <= tree_size; i++)
    {
        vec.push_back(i);
    }
    std::shuffle(vec.begin(), vec.end(), std::default_random_engine(seed));
    for (int i = 0; i < num_trees; i++)
    {
        eight_hundred(vec, max_heights, min_heights, tree_size);
        if(use_next_permutation)
            std::next_permutation(vec.begin(), vec.end());
        else 
            std::shuffle(vec.begin(), vec.end(), rng);
    }

    double max_height_sum = std::reduce(max_heights.begin(), max_heights.end());
    double min_height_sum = std::reduce(min_heights.begin(), min_heights.end());

    double average_max_height = max_height_sum / max_heights.size();
    double average_min_height = min_height_sum / min_heights.size();
    double average_total_height = calc_average_height(max_heights, min_heights, num_trees);

    std::vector<int>::iterator highest_max_height = std::max_element(max_heights.begin(), max_heights.end());
    std::vector<int>::iterator lowest_min_height = std::min_element(min_heights.begin(), min_heights.end());
    double average_difference = calc_average_diff_height(max_heights, min_heights, num_trees);
    double max_difference = calc_max_diff(max_heights, min_heights, num_trees);
    double min_difference = calc_min_diff(max_heights, min_heights, num_trees);


    std::cout << tree_size << " nodes in each tree. " << num_trees << " iterations." << std::endl;
    std::cout << "The average height of the tree in 800 iterations is: " << average_total_height << std::endl;
    std::cout << "The highest maximum height: " << *highest_max_height << std::endl;
    std::cout << "The average minimum height : " << average_min_height << std::endl;
    std::cout << "The lowest minimum height: " << *lowest_min_height << std::endl;
    std::cout << "The average difference between minimum and maximum height: " << average_difference << std::endl;
    std::cout << "The greatest difference between minimum and maximum height: " << max_difference << std::endl;
    std::cout << "The lowest difference between minimum and maximum height: " << min_difference << std::endl;
}

int main()
{   std::cout << "Use of next_permutation" << std::endl;
    standard_test(9000, 800, true);
    std::cout << "\n" << std::endl;
    std::cout << "Use of shuffle" << std::endl;
    standard_test(5000, 500, false);
}
