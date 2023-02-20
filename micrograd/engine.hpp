//
//  engine.hpp
//  Micrograd_C++
//
//  Created by Jacopo Zacchigna on 2023-02-19
//  Copyright © 2023 Jacopo Zacchigna. All rights reserved.
//

#pragma once

#include <array>
#include <iostream>

template <typename T> class Value {
public:
    T data;
    T grad;
    char label;
    std::array<std::shared_ptr<Value<T>>, 2> m_prev;
public:
    // Constructor
    Value(T data, char label = ' ', char op = ' ',
          std::array<std::shared_ptr<Value<T>>, 2> children = {nullptr,
                                                               nullptr})
        : data(data), label(label), m_op(op), m_prev(std::move(children)),
          grad(0) {}

    // Operator Overloading
    Value operator+(Value const &obj) const;
    Value operator-(Value const &obj) const;
    Value operator*(Value const &obj) const;
    Value operator/(Value const &obj) const;

    // << operator overload
    friend std::ostream &operator<<(std::ostream &os, const Value &v) {
        os << "Value(data=" << v.data << ", grad=" << v.grad << ")";
        return os;
    };

protected:
    char m_op;
};

template <typename T>
Value<T> Value<T>::operator+(Value<T> const &other) const {
    auto result =
        Value(data + other.data, ' ', '+',
              {std::make_shared<Value>(std::move(*this)), std::make_shared<Value>(std::move(other))});
    return result;
}
