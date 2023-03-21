#include <micrograd/nn.hpp>

int main() {

    // Three neurons
    auto neuron = Neuron<double>(3);

    std::vector<Value<double>> vec = {
        Value<double>(2.0, "first_value"),
        Value<double>(3.0, "second_value"),
        Value<double>(-7.0, "third_value")
    };

    auto x1 = std::make_shared<std::vector<Value<double>>>(vec);

    /* std::vector<Value<double>> x2 = { */
    /*         std::make_shared<Value<double>>(5.0, "first_value"), */
    /*         std::make_shared<Value<double>>(-8.0, "second_value"), */
    /*         std::make_shared<Value<double>>(3.0, "third_value"), */
    /* }; */

    // Testing the neuron output with two different set of values
    std::shared_ptr<Value<double>> y1 = neuron(x1);
    y1->backward();
    neuron.zero_grad();

    /* std::shared_ptr<Value<double>> y2 = neuron(x2); */
    /* y2->backward(); */
    /* neuron.zero_grad(); */

    std::cout << "Outputs:" << '\n';
    std::cout << "-----------------" << '\n';

    std::cout << "First pass: " << *y1 << '\n';
    /* // Neuron two should have rest the m_neurons value */
    /* std::cout << "Second pass: " << *y2 << "\n"; */

    std::cout << "Parameters: " << '\n';
    std::cout << "-----------------" << '\n';

    // Getting the neuron parameters
    for (auto &p : neuron.parameters()) {
        std::cout << *p << "\n";
    }

    y1->draw_graph();
}
