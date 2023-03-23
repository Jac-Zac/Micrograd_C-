#include <micrograd/nn.hpp>

#define SIZE 3
#define BATCH 4
typedef double TYPE;

inline Value_Vec<TYPE> read_dataset(const char *intput_file) {
    Value_Vec<TYPE> data;
    std::ifstream file(intput_file);
    if (!file.is_open()) {
        std::cout << "failed to open: " << intput_file << " file\n";
        return {};
    }

    double x;
    while (file >> x) {
        data.emplace_back(Value<TYPE>(x));
    }
    return data;
}


/*
void print_data(Value_Vec<TYPE> &inputs) {
    // input view it as a (2, x/2) instead of a (x)
    for(size_t i = 0; i < input.size(); i++){
        std::cout << input[i].data;
        if ( i %2 == 0){
            std::cout << ", ";
            continue;
        }
        std::cout << '\n';
    }

    // output
    for(auto value : target){
        std::cout << value.data << '\n';
    }
*/

inline std::vector<Value_Vec<TYPE>> forward(MLP<TYPE, 3> &model, Value_Vec<TYPE> &inputs) {

    std::vector<Value_Vec<TYPE>> scores;

    for (size_t i = 0; i < (inputs.size() - 1) ; i++) {
        // Forward pass
        /* scores.emplace_back(model(tmp)); */
        scores.emplace_back(model({inputs[i], inputs[i + 1]}));
    }
    return scores;
}

/*
Value<TYPE> loss(const std::vector<Value_Vec<TYPE>>& scores,
                                    const std:;vector<Value_Vec<TYPE>> y,
                                    const std::vector<Value<TYPE>>&
parameters) { Value_Vec<TYPE> losses;

  for (auto i = 0; i < y.size(); ++i) {
    losses.emplace_back(
        (make_shared<Value>(1.0) + (-y[i][0]) * scores[i][0])->relu());
  }
  // svm "max-margin" loss
  auto data_loss = std::accumulate(losses.begin(), losses.end(),
make_shared<Value>(0.0)); data_loss = data_loss /
make_shared<Value>(losses.size());

  // L2 regularization
  auto alpha = make_shared<Value>(1e-4);
  auto square_sum = std::inner_product(parameters.begin(), parameters.end(),
    parameters.begin(), make_shared<Value>(0.0));
  auto reg_loss = alpha * square_sum;
  auto total_loss = data_loss + reg_loss;

  // get accuracy
  double accuracy = 0.0;
  for (auto i = 0; i < y.size(); ++i) {
    accuracy += (scores[i][0]->data() > 0) == (y[i][0]->data() > 0);
  }
  accuracy /= y.size();
  return std::make_tuple(total_loss, accuracy);
}

*/

int main(int argc, char *argv[]) {

    if (argc < 3) {
        std::cout << "Usage: mlp_example X.txt y.txt\n";
        return -1;
    }

    // Need to find the files
    Value_Vec<TYPE> inputs = read_dataset(argv[1]);
    Value_Vec<TYPE> target = read_dataset(argv[2]);

    // SIZE is equal to the number of layers without the first one
    auto model = MLP<TYPE, SIZE>(2, {16, 16, 1});
    std::cout << model << '\n';
    std::cout << "number of parameters: " << model.parameters().size() << "\n";

    const size_t epochs = 100;
    for (size_t epoch = 0; epoch < epochs; ++epoch) {

        auto scores = forward(model, inputs);
        /* scores[0][0].draw_graph(); */

        break;
        /*
        auto [total_loss, acc] = loss(scores, y, model.parameters());

        model.zero_grad();
        total_loss->backward();

        double learning_rate = 1.0 - 0.9 * epoch / 100;
        learning_rate = std::max(learning_rate, 0.001);

        for (auto p : model.parameters()) {
            p->_data -= learning_rate * p->grad();
        }

        std::cout << "epoch: " << epoc << " loss: " << total_loss->data
                  << ", accuracy " << acc * 100 << ", lr: "<< learning_rate << "\n ";
        */
    }
}
