//  engine.hpp
//  Micrograd_C++
//
//  Created by Jacopo Zacchigna on 2023-02-19
//  Copyright © 2023 Jacopo Zacchigna. All rights reserved.

#pragma once

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace value_engine {

enum ops_type : char {
    ADD = '+',
    DIF = '-',
    MUL = '*',
    DIV = '/',
    POW = '^',
    INV = 'i',
    EXP = 'e',
    NEG = 'n',
    TANH = 't',
    RELU = 'r',
    LRELU = 'l',
    SWISH = 's'
};

template <typename T> class Value {
public:
    std::string label; // label of the value
    T data;            // data of the value
    T grad;            // gradient which by default is zero

    /* protected: */
protected:
    char m_op;
    std::array<Value<T> *, 2> m_prev; // previous values
    std::vector<Value<T> *>
        m_sorted_values; // vector to store the sorted values
    std::unordered_set<Value<T> *> m_visited; // keep track of the visited nodes
public:
    // Constructor
    Value(T data, std::string label = "", char op = ' ',
          std::array<Value<T> *, 2> children = {nullptr, nullptr})
        : data(data), label(label), m_op(op), m_prev(std::move(children)),
          grad(0.0) {}

    // Operator Overloading
    // lvalues and rvalues because of const reference
    friend Value operator+(const Value &lhs, const Value &rhs) {
        return Value(lhs.data + rhs.data, "", ADD,
                     {const_cast<Value *>(&lhs), const_cast<Value *>(&rhs)});
    }

    friend Value operator-(const Value &lhs, const Value &rhs) {
        return Value(lhs.data - rhs.data, "", DIF,
                     {const_cast<Value *>(&lhs), const_cast<Value *>(&rhs)});
    }

    friend Value operator*(const Value &lhs, const Value &rhs) {
        return Value(lhs.data * rhs.data, "", MUL,
                     {const_cast<Value *>(&lhs), const_cast<Value *>(&rhs)});
    }

    friend Value operator/(const Value &lhs, const Value &rhs) {
        return Value(lhs.data / rhs.data, "", DIV,
                     {const_cast<Value *>(&lhs), const_cast<Value *>(&rhs)});
    }

    friend Value operator^(const Value &lhs, const Value &rhs) {
        return Value(std::pow(lhs.data, rhs.data), "", POW,
                     {const_cast<Value *>(&lhs), const_cast<Value *>(&rhs)});
    }

    friend Value operator+=(Value &lhs, const Value &rhs) {
        // std::shared_ptr<Value<T>> tmp1 = std::make_shared<Value<T>>(lhs.data,
        // "", lhs.m_op, lhs.m_prev); std::shared_ptr<Value<T>> tmp2 =
        // std::make_shared<Value<T>>(rhs.data, "", rhs.m_op, rhs.m_prev);

        Value<T> *tmp1 =
            new Value<T>(lhs.data, lhs.label, lhs.m_op, lhs.m_prev);
        Value<T> *tmp2 =
            new Value<T>(rhs.data, rhs.label, rhs.m_op, rhs.m_prev);

        if (rhs.m_prev[0] == nullptr) {
            lhs = *tmp1 + rhs;
            delete tmp2;
        }
        // Create a copy as usual
        else {
            lhs = *tmp1 + *tmp2;
        }

        return lhs;
    }

    // << operator overload
    friend std::ostream &operator<<(std::ostream &os, const Value<T> &v) {
        os << std::fixed;
        os << std::setprecision(16);
        os << "Value(data=" << v.data << ", grad=" << v.grad
           << ", label=" << v.label << ")";
        return os;
    };

    // Other ops
    Value<T> inverse_value();
    Value<T> exp_value();
    Value<T> tanh();
    Value<T> relu();
    Value<T> lrelu();
    Value<T> swish();

    void backward();
    void draw_graph();

protected:
    // Helper function to make a topological sort
    void _topo_sort(Value<T> *v);
    void _backward_single(); // 1 step of backdrop
    // Function to update the gradient
    void _update_grad(T grad) { this->grad += grad; }
};

// Adding aliases
template <typename T> using Value_Vec = std::vector<Value<T>>;
template <typename T>
using Value_Vec_Ptr = std::vector<std::shared_ptr<Value<T>>>;
template <typename T>
using Ptr_Value_Vec = std::shared_ptr<std::vector<Value<T>>>;

// ==================== Implementation =====================

template <typename T> Value<T> Value<T>::inverse_value() {
    return Value(1.0 / this->data, "", INV, {this, nullptr});
}

template <typename T> Value<T> Value<T>::exp_value() {
    return Value<T>(std::exp(data), "", EXP, {this, nullptr});
}

template <typename T> Value<T> Value<T>::tanh() {
    return Value<T>(std::tanh(data), "", TANH, {this, nullptr});
}

template <typename T> Value<T> Value<T>::relu() {
    return Value<T>(data < 0.0 ? 0.0 : data, "", RELU, {this, nullptr});
}

template <typename T> Value<T> Value<T>::lrelu() {
    return Value<T>(data > 0.0 ? data : 0.01 * data, "", LRELU,
                    {this, nullptr});
}

template <typename T> Value<T> Value<T>::swish() {
    // swish = x * sigmoid(x)
    // sigmoid = 1/(1 + e^-x)
    return Value<T>(data / (1.0 + std::exp(-data)), "", SWISH, {this, nullptr});
}

template <typename T> void Value<T>::_backward_single() {
    switch (this->m_op) {
    case ADD:
        // Should just move the gradient along to both of them
        // += because we want to avoid bugs if we reuse a variable
        m_prev[0]->_update_grad(this->grad);
        m_prev[1]->_update_grad(this->grad);
        break;
    case DIF:
        // same as m_prev[0] += 1.0 * grad;
        m_prev[0]->_update_grad(grad);
        m_prev[1]->_update_grad(-grad); // same as doing -=
        break;
    case MUL:
        // same as m_prev[0] += m_prev[1]->data * grad
        m_prev[0]->_update_grad(m_prev[1]->data * grad);
        m_prev[1]->_update_grad(m_prev[0]->data * grad);
        break;
    case DIV:
        m_prev[0]->_update_grad((1 / (m_prev[1]->data)) * grad);
        m_prev[1]->_update_grad(-(m_prev[0]->data) /
                                std::pow(m_prev[1]->data, 2) * grad);
        break;
    case POW:
        m_prev[0]->_update_grad(
            (m_prev[1]->data *
             std::pow(m_prev[0]->data, (m_prev[1]->data - 1))) *
            grad);
        break;
    case INV:
        // e^x is e^x which I already saved in data
        m_prev[0]->_update_grad(-1 / std::pow(m_prev[0]->data, 2.0) * grad);
        break;
    case EXP:
        // e^x is e^x which I already saved in data
        m_prev[0]->_update_grad(data * grad);
        break;
    case TANH:
        m_prev[0]->_update_grad((1 - std::pow(data, 2)) * grad);
        break;
    case RELU:
        m_prev[0]->_update_grad(data > 0.0 ? (1.0 * grad) : 0.0);
        break;
    case LRELU:
        m_prev[0]->_update_grad(data > 0.0 ? (1.0 * grad) : 0.01 * grad);
        break;
    case SWISH:
        // keep in mind that data = swish(m_prev[0]->data)
        // and f'(x) = f(x) + sigmoid(x)(1 + f(x))
        m_prev[0]->_update_grad(
            (data +
             ((1.0 / (1.0 + std::exp(-m_prev[0]->data))) * (1.0 + data))) *
            grad);
        break;
    default:
        break;
    }
}

template <typename T> void Value<T>::_topo_sort(Value<T> *v) {
    // Add it to the visited values
    m_visited.insert(v);

    // Iterate trough the children
    for (auto *child : v->m_prev) {
        // If not visited and not a leaf value
        if (m_visited.count(child) == 0 && child != nullptr) {
            // Call the function recursively
            _topo_sort(child);
        }
    }
    m_sorted_values.push_back(v);
}

template <typename T> void Value<T>::backward() {

    // If empty do topo sort
    if (m_sorted_values.empty()) {
        _topo_sort(this);
    }

    // Set the derivative of dx/dx to 1
    this->grad = 1.0;

    // Call backward in topological order applying the chain rule automatically
    for (auto it = m_sorted_values.rbegin(); it != m_sorted_values.rend();
         ++it) {
        (*it)->_backward_single();
    }
}

template <typename T> void Value<T>::draw_graph() {
    // Perform a topological sort of the graph in reverse
    if (m_sorted_values.empty()) {
        _topo_sort(this);
    }

    // Open a file to write the output
    std::ofstream outfile("graph.dot");
    if (!outfile) {
        std::cerr << "Error: failed to open graph.dot\n";
        return;
    }

    // Create a graphviz graph
    outfile << "digraph G {\n";
    outfile << "  rankdir=LR; // set rankdir attribute to LR\n";
    for (const auto &values : m_sorted_values) {
        outfile << "  " << uintptr_t(values)
                << " [label=\"label = " << values->label
                << " | data = " << values->data << " | grad = " << values->grad
                << "\", shape=record]\n";
        if (values->m_op != ' ') {
            // if this value is a result of some operation, create an op node
            // for it
            outfile << "  " << uintptr_t(values) + values->m_op << " [label=\""
                    << values->m_op << "\"]\n";
            outfile << "  " << uintptr_t(values) + values->m_op << " -> "
                    << uintptr_t(values) << "\n";
        }
        for (size_t j = 0; j < 2; j++)
            if (values->m_prev[j] != nullptr) {
                // if this value is a result of some operation, create an op
                // node for it
                outfile << "  " << uintptr_t(values->m_prev[j]) << " -> "
                        << uintptr_t(values) + values->m_op << "\n";
            }
    }

    outfile << "}\n";
    outfile.close();

    // Create the graph using the dot command
    std::system("dot -Tsvg graph.dot -o graph.svg");
    // Open the graph using the default viewer
    std::system("open graph.svg");
}
} // namespace value_engine
