#include <micrograd/nn.hpp>

#define SIZE 3
#define BATCH 4
/* #define DATASET_SIZE 3 * 4 */

typedef double TYPE;

int main() {
    // Binary classification

    // define the neural network
    std::array<size_t, SIZE> n_neurons_for_layer = {4, 4, 1};
    auto model = MLP<TYPE, SIZE>(3, n_neurons_for_layer);

    // Create a vector for the input and view it as a (3, 4)
    /* Value_Vec<TYPE> xs = {2.0, 3.0, -1.0, */
    /*                       3.0, -1.0, 0.5, */
    /*                       0.5, 1.0, 1.0, */
    /*                       1.0, 1.0, -1.0}; */

    std::vector<Value_Vec<TYPE>> xs = {
        {2.0, 3.0, -1.0}, {3.0, -1.0, 0.5}, {0.5, 1.0, 1.0}, {1.0, 1.0, -1.0}};


    /* Value_Vec_Ptr<TYPE> xs = std::make_shared<TYPE>( */
    /*         2.0, 3.0, -1.0, */
    /*         3.0, -1.0, 0.5, */
    /*         0.5, 1.0, 1.0, */
    /*         1.0, 1.0, -1.0); */

    // desired target
    Value_Vec<TYPE> ys = {1.0, -1.0, -1.0, 1.0};

    std::cout << model; // to output the network shape

    std::cout << "\nThe network has: " << model.parameters().size()
              << " parameters\n\n";

    std::cout << "Starting Training\n";
    std::cout << "----------------------------\n\n";

    double lr = 0.005;

    for (size_t j = 1; j <= 1000; j++) {

        // Zero grad
        model.zero_grad();

        auto loss = model.MSE_loss_backprop(xs, ys, BATCH);

        // Change the learning rate
        if (j < 800) {
            lr = 0.005;
        } else {
            lr = 0.001;
        }

        // Update parameters thanks to the gradient
        for (Value<TYPE> *p : model.parameters()) {
            // Update parameter value
            p->data += -0.01 * (p->grad);
        }

        if (j % 100 == 0) {
            std::cout << "The loss at step: " << j << " is: " << loss.data
                      << '\n';
        }
    }
}
