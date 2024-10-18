//
// DD1388 - Lab 9: A board game
//
// Author: Ingemar Markstrom, ingemarm@kth.se
//

#ifndef VEC2D_HPP
#define VEC2D_HPP

// Coming from the world of 2/3D graphics, enjoy this simple yet
// useful 2D vector implementation. A bit overkill mayhaps, but it
// helps with readability later, as well as conceptualizing.
// Accepts any regular numeric types, int/float/doubles etc.

template <class T>
struct Vec2d {
    T x;
    T y;
    Vec2d() : x{}, y{} {}
    Vec2d(T xx, T yy) : x(xx), y(yy) {}

    Vec2d<T> operator+(const Vec2d<T> & o) const;
    Vec2d<T> operator-(const Vec2d<T> & o) const;
    Vec2d<T> operator*(const Vec2d<T> & o) const;

    template <class S>
    Vec2d<T> operator*(const S & i ) const ;
  
    Vec2d<bool> operator==(const Vec2d<T> & o) const ;
    bool operator==(const bool & o) const ;
    Vec2d<bool> operator<(const Vec2d<T> & o) const ;
    Vec2d<bool> operator<=(const Vec2d<T> & o) const ;
    Vec2d<bool> operator&&(const Vec2d<T> & o) const ;
    // This one is actually pretty neat. Think about it for a second... :)
    // Why is this one so useful?
    operator bool() const ;
};

//---------------------------------------------------------

template <class T>
Vec2d<T> Vec2d<T>::operator+(const Vec2d<T> & o) const {
    return { x + o.x, y + o.y };
}
template <class T>
Vec2d<T> Vec2d<T>::operator-(const Vec2d<T> & o) const {
    return { x - o.x, y - o.y };
}
template <class T>
Vec2d<T> Vec2d<T>::operator*(const Vec2d<T> & o) const {
    return { x * o.x, y * o.y };
}
template <class T>
template <class S>
Vec2d<T> Vec2d<T>::operator*(const S & i ) const {
    return { x * i, y * i };
}
template <class T>
Vec2d<bool> Vec2d<T>::operator==(const Vec2d<T> & o) const {
    return { x == o.x, y == o.y };
}
template <class T>
bool Vec2d<T>::operator==(const bool & o) const {
    return (x == o) && (y == o);
}
template <class T>
Vec2d<bool> Vec2d<T>::operator<(const Vec2d<T> & o) const {
    return { x < o.x , y < o.y };
}
template <class T>
Vec2d<bool> Vec2d<T>::operator<=(const Vec2d<T> & o) const {
    return { x <= o.x , y <= o.y };
}
template <class T>
Vec2d<bool> Vec2d<T>::operator&&(const Vec2d<T> & o) const {
    return { x && o.x , y && o.y };
}

template <class T>
Vec2d<T>::operator bool() const { return !( x == T{} || y == T{} ); }

#endif  // VEC2D_HPP