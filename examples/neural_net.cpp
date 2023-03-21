#include <micrograd/nn.hpp>

#define N_OUTPUT_LAYERS 3

int main() {
    /// Initialize the neural network

    std::array<size_t, N_OUTPUT_LAYERS> output_layers_sizes = {4, 4, 1};
    auto model = MLP<double, N_OUTPUT_LAYERS>(2, output_layers_sizes);

    std::vector<Value<double>> x = {
        Value<double>(2.0, "first_value"),
        Value<double>(3.0, "second_value"),
    };

    /* std::vector<double> x = {2.0,3.0}; */

    // auto will be an std::variant
    /* std::vector<Value_Vec<double>> y = model(x); */
    auto y = model(x);

    // We access the last of the of the output layers since it is 1 - N_TOTAL
    // which is perfect to access the vector 0 because we only have 1 neuron
    y[N_OUTPUT_LAYERS][0].backward();
    /* model.zero_grad(); */

    for (auto &p : model.parameters()) {
        std::cout << *p << "\n";
    }

    std::cout << "Model information: " << model << "\n";
    std::cout << "Number of parameters: " << model.parameters().size() << "\n";

    std::cout << "Output: " << y[N_OUTPUT_LAYERS][0] << "\n";

    y[N_OUTPUT_LAYERS][0].draw_graph();

    return 0;
}
