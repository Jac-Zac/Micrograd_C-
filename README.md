# microcpp_grad

<!-- ![awww](puppy.jpg) -->

A tiny Autograd engine rewritten in c++ from [`micrograd`](https://github.com/karpathy/micrograd). Implements back-prop (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API. Both are tiny, with about 260 and 230 lines of code respectively. The DAG only operates over scalar values, so e.g. we chop up each neuron into all of its individual tiny adds and multiplies. However, this is enough to build up entire deep neural nets doing binary classification.

### Installation
> Change your default svg viewer to your browser if you want to see render

```bash
cmake -Boutput && cd output && make && ./test_executable
```

### Single perception example;
> Example of a perception to show different ops
```cpp
#include "../micrograd/engine.hpp"

int main() {
    // Creating a single perception

    // Input x1, x2
    auto x1 = Value<double>(2.0, "x1"), x2 = Value<double>(0.0, "x2");
    // Weight w1, w2
    auto w1 = Value<double>(-3.0, "w1"), w2 = Value<double>(1.0, "w2");

    // products
    auto x1w1 = x1 * w1;
    x1w1.label = "x1*w1";
    auto x2w2 = x2 * w2;
    x2w2.label = "x2*w2";

    // sum of the two
    auto x1w1_x2w2 = x1w1 + x2w2;
    x1w1_x2w2.label = "x1w1 + x2w2";

    // Bias of the neuron b
    auto b = Value<double>(6.881375870, "b");

    // new neuron
    auto n = (x1w1_x2w2 + b);
    n.label = "n";

    // auto o = n.tanh();

    // Custom tanh implementation
    auto e = (n * 2).exp_value();
    e.label = "e";
    auto o = (e - 1) / (e + 1);
    o.label = "o";

    // Grandina with respect to itself is 1
    o.backward();
    o.draw_graph();
}
```

### TODO

- [ ] Try using shared pointers

- [ ] Think of a better way to write the autograd engine. And also stack based topo_sort

- Think about mini batches because when you mini batch you compute less accurate gradients for loss minimization but you can do it way faster and thus do way more step
