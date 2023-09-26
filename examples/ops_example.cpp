//  ops_example.cpp
//  micrograd_c++
//
//  created by jacopo zacchigna on 2023-02-19
//  copyright © 2023 jacopo zacchigna. all rights reserved.

#include <micrograd/engine.hpp>

using namespace value_engine;

int main() {
    // Testing
    auto a = Value<double>(-4.0, "a");
    auto b = Value<double>(2.0, "b");
    auto c = a + b;
    auto d = a * b + (b ^ 3);
    // Need to create some support variable for the += operation
    auto mid1 = c;
    mid1.label = "mid";
    c += mid1 + 1;
    c.label = "c";
    auto mid2 = c;
    c += 1 + mid2 - a;
    auto tmp1 = d;
    d += tmp1 * 2 + (b + a).relu();
    auto tmp2 = d;
    d += 3 * tmp2 + (b - a).relu();
    d.label = "d";
    auto e = c - d;
    e.label = "e";
    auto f = e ^ 2;
    f.label = "f";
    auto g = (f / 2.0);
    g.label = "g";
    g += f.inverse_value() * 10;
    g.backward();
    g.draw_graph();
    std::cout << g << '\n';
    std::cout << a << '\n';
    std::cout << b << '\n';
}
