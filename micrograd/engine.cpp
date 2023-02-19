//
//  engine.cpp
//  Micrograd_C++
//
//  Created by Jacopo Zacchigna on 2023-02-19
//  Copyright © 2023 Jacopo Zacchigna. All rights reserved.
//

#include "engine.hpp"

// Implementation of the constructor which initialize and empty m_prev and zero
// gradient Constructor initialize m_prev with the children of the previous
// Value and also the operator they had if they are not given uses default values
template <typename T>
Value<T>::Value(T data, char label, char op, std::array<Value<T>,2> children) : data(data), grad(0), m_prev({op, children}), label(label)
{
    // self.m_bacward =
}

template <typename T>
Value<T> Value<T>::operator+(const Value<T> &other) const {
    Value<T> result = Value(data + other.data);
    result.m_prev = { '+', { this, &other }};

    /* auto backward = [](Value<T> &result) { */
    /*     result.m_op1->grad += result.grad; */
    /*     result.m_op2->grad += result.grad; */
    /* }; */

    /* result.set_backward(backward); */

    // Using move semantic to avoid unnecessary coping by value
    return std::move(result);
}
