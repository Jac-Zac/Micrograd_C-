#include <micrograd/nn.hpp>

int main() {

    // Three neurons
    auto neuron = Neuron<double>(3);

    std::vector<Value<double>> x1 = {Value<double>(2.0, "first_value"),
                                     Value<double>(3.0, "second_value"),
                                     Value<double>(-7.0, "third_value")};

    std::vector<Value<double>> x2 = {Value<double>(5.0, "first_value"),
                                     Value<double>(-8.0, "second_value"),
                                     Value<double>(3.0, "third_value")};

    // Testing the neuron output with two different set of values
    Value<double> y1 = neuron(x1);
    y1.backward();
    /* neuron.zero_grad(); */

    auto y2 = neuron(x2);
    y2.backward();

    std::cout << "Outputs:" << '\n';
    std::cout << "-----------------" << '\n';

    std::cout << "First pass: " << y1 << '\n';
    // Neuron two should have rest the m_neurons value
    std::cout << "Second pass: " << y2 << "\n";

    std::cout << "Parameters: " << '\n';
    std::cout << "-----------------" << '\n';

    // Getting the neuron parameters
    for (auto &p : neuron.parameters()) {
        std::cout << *p << "\n";
    }

    y2.draw_graph();
}
