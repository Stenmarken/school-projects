#ifndef COMPLEX_H
#define COMPLEX_H

#include <initializer_list>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <math.h>


class Complex
{
public:
    constexpr Complex();
    constexpr Complex(double real);
    constexpr Complex(double real, double imaginary);
    constexpr Complex(const Complex &rhs);
    Complex &operator=(const Complex &other);
    void operator+=(const Complex &other);
    void operator-=(const Complex &other);
    void operator*=(const Complex &other);
    void operator/=(const Complex &other);
    double real() const;
    double imag() const;

private:
    double real_part;
    double imaginary_part;
};

constexpr Complex::Complex(): 
    real_part(0), imaginary_part(0) {}

constexpr Complex::Complex(double real) : 
    real_part(real), imaginary_part(0) {}

constexpr Complex::Complex(double real, double imaginary) :
    real_part(real), imaginary_part(imaginary) {}

constexpr Complex::Complex(const Complex &rhs) :
    real_part(rhs.real_part), imaginary_part(rhs.imaginary_part) {}

Complex &Complex::operator=(const Complex &other)
{
    real_part = other.real_part;
    imaginary_part = other.imaginary_part;
    return *this;
}

void Complex::operator+=(const Complex &other)
{
    real_part += other.real_part;
    imaginary_part += other.imaginary_part;
}

void Complex::operator-=(const Complex &other)
{
    real_part -= other.real_part;
    imaginary_part -= other.imaginary_part;
}

void Complex::operator*=(const Complex &other)
{
    double old_real_part = real_part;
    real_part = real_part * other.real_part - imaginary_part * other.imaginary_part;
    imaginary_part = (old_real_part * other.imaginary_part + imaginary_part * other.real_part);
}

void Complex::operator/=(const Complex &other)
{
    double old_real_part = real_part;
    real_part = (real_part * other.real_part + imaginary_part * other.imaginary_part) / 
                (pow(other.real_part, 2) + pow(other.imaginary_part, 2));

    imaginary_part = (imaginary_part * other.real_part - old_real_part * other.imaginary_part) / 
                     (pow(other.real_part, 2) + pow(other.imaginary_part, 2));
}

double Complex::real() const
{
    return real_part;
}

double Complex::imag() const
{
    return imaginary_part;
}

Complex operator+(const Complex &c1)
{
    return c1;
}

Complex operator-(const Complex &c1)
{
    Complex c2(-c1.real(), -c1.imag());
    return c2;
}

// Addition for two complex numbers
Complex operator+(const Complex &c1, const Complex &c2)
{
    Complex c3(c1.real() + c2.real(), c1.imag() + c2.imag());
    return c3;
}
// (a + bi) + c = a + c + bi
Complex operator+(const Complex &c1, const double c2)
{
    Complex c3(c1.real() + c2, c1.imag());
    return c3;
}
// c + (a + bi) = a + c + bi
Complex operator+(const double c1, const Complex &c2)
{
    Complex c3(c2.real() + c1, c2.imag());
    return c3;
}

// Subtraction for two complex numbers
Complex operator-(const Complex &c1, const Complex &c2)
{
    Complex c3(c1.real() - c2.real(), c1.imag() - c2.imag());
    return c3;
}

// (a + bi) - c = a - c + bi
Complex operator-(const Complex &c1, const double c2)
{
    Complex c3(c1.real() - c2, c1.imag());
    return c3;
}

// c - (a + bi) = c - a - bi
Complex operator-(const double c1, const Complex &c2)
{
    Complex c3(c1 - c2.real(), -c2.imag());
    return c3;
}

// Multiplication for two complex numbers
Complex operator*(const Complex &c1, const Complex &c2)
{
    Complex c3(c1.real() * c2.real() - c1.imag() * c2.imag(), 
               c1.real() * c2.imag() + c1.imag() * c2.real());
    return c3;
}

// (a + bi) * c = ac + bci
Complex operator*(const Complex &c1, const double c2)
{
    Complex c3(c1.real() * c2, c1.imag() * c2);
    return c3;
}

// c * (a + bi) = ac + bci
Complex operator*(const double c1, const Complex &c2)
{
    Complex c3(c2.real() * c1, c2.imag() * c1);
    return c3;
}

// Division for two complex numbers
Complex operator/(const Complex &c1, const Complex &c2)
{
    Complex c3((c1.real() * c2.real() + c1.imag() * c2.imag()) / 
               (pow(c2.real(), 2) + pow(c2.imag(), 2)), 
               (c1.imag() * c2.real() - c1.real() * c2.imag()) / 
               (pow(c2.real(), 2) + pow(c2.imag(), 2)));
    return c3;
}

// (a + bi) / c = a/c + bci/c
Complex operator/(const Complex &c1, const double c2)
{
    return operator/(c1, Complex(c2, 0));
    //Complex c3(c1.real() / c2, c1.imag() / c2);
    //return c3;
}

// c / (a + bi) = ca/(a^2 + b^2) + cbi/(a^2 + b^2)
Complex operator/(const double c1, const Complex &c2)
{
    return operator/(Complex(c1, 0), c2);
    /*Complex c3(c1 * c2.real() / (pow(c2.real(), 2) + pow(c2.imag(), 2)), 
               c1 * c2.imag() / (pow(c2.real(), 2) + pow(c2.imag(), 2)));*/
}

bool operator==(const Complex &c1, const Complex &c2)
{
    return (c1.real() == c2.real() && c1.imag() == c2.imag());
}

bool operator!=(const Complex &c1, const Complex &c2)
{
    return !(c1 == c2);
}

bool operator==(const Complex &c1, const double c2)
{
    return (c1.real() == c2 && c1.imag() == 0);
}

bool operator!=(const Complex &c1, const double c2)
{
    return !(c1 == c2);
}

bool operator==(const double c1, const Complex &c2)
{
    return (c2 == c1);
}

bool operator!=(const double c1, const Complex &c2)
{
    return !(c2 == c1);
}

double real(const Complex &c)
{
    return c.real();
}

double imag(const Complex &c)
{
    return c.imag();
}

double abs(const Complex &c)
{
    return sqrt(pow(c.real(), 2) + pow(c.imag(), 2));
}

bool operator<(const Complex &c1, const Complex &c2)
{
    return abs(c1) < abs(c2);
}

std::ostream &operator<<(std::ostream &out, const Complex &c)
{
    out << "(" << c.real() << ", " << c.imag() << "i" << ")";
    return out;
}

std::istream &operator>>(std::istream &in, Complex &c)
{
    std::string s;
    double real;
    double imag;
    char par = '(';
    in >> s;
    if(s[0] == par)
    {
        s.erase(0, 1);
        char c = ',';
        int index = s.find(c);
        if(index != std::string::npos)
        {
            std::string sub = std::string(&s[0], &s[index]);
            real = std::stof(sub);
            s.erase(0, index+1);
            sub.pop_back();
            imag = std::stof(s);
        }
        else
        {
            s.pop_back();
            real = std::stof(s);
            imag = 0;
        }
    }
    else 
    {
        real = std::stof(s);
        imag = 0;
    } 
    c = Complex(real, imag);
    return in;
}

constexpr Complex operator ""_i(long double d) 
{
    return Complex(0, d);
}

constexpr Complex operator ""_i(unsigned long long int d)
{
    return Complex(0, d);
}



#endif // COMPLEX_H