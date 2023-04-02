//  nn.hppT
//  Micrograd_C++
//
//  Created by Jacopo Zacchigna on 2023-02-19
//  Copyright © 2023 Jacopo Zacchigna. All rights reserved.

#pragma once

#include "engine.hpp"
#include <random>
/* #include <variant> */

using namespace value_engine;

template <typename T> T random_uniform(T range_from, T range_to) {
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::uniform_real_distribution<T> distr(range_from, range_to);
    return distr(generator);
}

// ---------------------------------------------------------

// Module Parent class as an interface
template <typename T> class Module {
public:
    void zero_grad() {
        for (auto &p : parameters()) {
            p->grad = 0.0;
        }
        m_weighted_sums.clear();
        m_layers_output.clear();
    }
    // Make it virtual so that it can be override
    virtual std::vector<Value<T> *> parameters() { return {}; }

protected:
    Value_Vec_Ptr<T> m_weighted_sums;
    std::vector<Ptr_Value_Vec<T>> m_layers_output;
};

template <typename T> class Neuron : public Module<T> {
public:
    Neuron(size_t num_neurons_input, bool nonlin = true);
    virtual ~Neuron(){};

    // Call operator: w * x + b dot product
    Value<T> operator()(const Value_Vec<T> &x);

    // Overriding
    virtual std::vector<Value<T> *> parameters() override;

protected:
    size_t m_num_neurons_input;
    bool m_nonlin;
    std::vector<Value<T>> m_weights;
    // I'm not propagating the gradient to the bias
    Value<T> m_bias;
    using Module<T>::m_weighted_sums; // bring m_weighted_sums into the scope of
                                      // Neuron
};

// ---------------------------------------------------------

template <typename T> class Layer : public Module<T> {
public:
    Layer(size_t num_neurons_input, size_t num_neurons_out, bool nonlin = true);
    virtual ~Layer(){};

    // Call operator: forward for every neuron in the layer
    Value_Vec<T> operator()(const Value_Vec<T> &x);

    // Overriding
    virtual std::vector<Value<T> *> parameters() override;

protected:
    // Create the neurons for the layer
    std::vector<Neuron<T>> m_neurons;
};

// ----------------------------------------------------------

template <typename T, size_t N> class MLP : public Module<T> {
public:
    MLP(size_t num_neurons_input, std::array<size_t, N> num_neurons_out);
    virtual ~MLP(){};

    // Call operator: w * x + b dot product
    Value_Vec<T> operator()(const Value_Vec<T> &x);

    Value<T> MSE_loss_backprop(std::vector<Value_Vec<T>> &input,
                               std::vector<Value_Vec<T>> &target, size_t batch_size);

    // Declare the operator<< function as a friend function and get the
    // structure of the network
    friend std::ostream &operator<<(std::ostream &os, const MLP<T, N> &mlp) {
        os << "Network of " << N + 1 << " Layers: [ " << mlp.m_num_neurons_in;
        for (size_t i = 0; i < N; i++) {
            os << " , " << mlp.m_num_neurons_out[i];
        }
        os << " ]\n";
        return os;
    }

    // Overriding
    virtual std::vector<Value<T> *> parameters() override;

public:
    std::vector<Layer<T>> m_layers;
    size_t m_net_size;
    // Layer given in output
protected:
    const size_t m_num_neurons_in;
    // N layers of the N + 1 total have outputs
    const std::array<size_t, N> m_num_neurons_out;
    using Module<T>::m_layers_output; // bring m_layers_output into the scope of
                                      // Neuron
};

//  ================ Implementation  Neuron =================

template <typename T>
Neuron<T>::Neuron(size_t number_of_neurons_input, bool nonlin)
    : m_num_neurons_input(number_of_neurons_input), m_nonlin(nonlin),
      /* m_bias(Value<T>(random_uniform(-1.0, 1.0), "bias")) { */
      m_bias(Value<T>(0.0, "bias")) {
    for (size_t i = 0; i < m_num_neurons_input; i++) {
        m_weights.emplace_back(Value<T>(random_uniform(-1.0, 1.0), "weight"));
    }
}

// This can perhaps just take an in input std::vector<T> and not a vector of
// values
template <typename T> Value<T> Neuron<T>::operator()(const Value_Vec<T> &x) {

    m_weighted_sums.push_back(std::make_shared<Value<T>>(0.0));

    // Sum over all multiplies
    for (size_t i = 0; i < m_num_neurons_input; i++) {
        // -> doesn't work and I do not know why
        *m_weighted_sums.back() += m_weights[i] * x[i];
    }

    // Add the bias
    *m_weighted_sums.back() += m_bias;

    // return the activated value
    /* return m_nonlin ? m_weighted_sums.back()->relu() :
     * *m_weighted_sums.back(); */
    return m_nonlin ? m_weighted_sums.back()->lrelu() : *m_weighted_sums.back();
    /* return m_nonlin ? m_weighted_sums.back()->tanh() :
     * *m_weighted_sums.back(); */

    /* return m_nonlin ? m_weighted_sums.back()->swish()
     * :*m_weighted_sums.back(); */
}

template <typename T> std::vector<Value<T> *> Neuron<T>::parameters() {
    // Create a vector for the pointers to the parameters to modici them
    // directly
    std::vector<Value<T> *> params;

    // Add the biases
    params.push_back(&m_bias);

    for (auto &w : m_weights) {
        // Add the weights
        params.push_back(&w);
    }
    return params;
}

//  ================ Implementation  Layer =================

template <typename T>
Layer<T>::Layer(size_t num_neurons_input, size_t num_neurons_output,
                bool nonlin) {
    // Add all the neurons to the layer by crating them
    for (size_t i = 0; i < num_neurons_output; i++) {
        m_neurons.emplace_back(Neuron<T>(num_neurons_input, nonlin));
    }
}

template <typename T> Value_Vec<T> Layer<T>::operator()(const Value_Vec<T> &x) {

    // Create a vector of Value objects to return
    Value_Vec<T> m_neurons_output;

    // Iterate over the neurons and push the result of calling each neuron
    for (auto &neuron : m_neurons) {
        m_neurons_output.emplace_back(neuron(x));
    }
    return m_neurons_output;
}

template <typename T> std::vector<Value<T> *> Layer<T>::parameters() {
    std::vector<Value<T> *> params;
    // Iterate over all the neurons
    for (auto &neuron : m_neurons) {
        auto neuron_params = neuron.parameters();
        // insert an object at the end thus we are not using emplace_back
        params.insert(params.end(), neuron_params.begin(), neuron_params.end());
    }
    return params;
}

//  ================ Implementation MLP =================

template <typename T, size_t N>
MLP<T, N>::MLP(size_t num_neurons_input,
               std::array<size_t, N> num_neurons_output)
    : m_num_neurons_in(num_neurons_input),
      m_num_neurons_out(num_neurons_output) {

    // Create the first layer with the input neuron size
    m_layers.emplace_back(Layer<T>(num_neurons_input, num_neurons_output[0]));

    // Create the following layers
    for (size_t i = 1; i < N; i++) {
        // Create layers N layers with the number of neuron from the previous
        // layers and output as the current
        bool nonlin = (i != N - 1);
        m_layers.emplace_back(
            Layer<T>(num_neurons_output[i - 1], num_neurons_output[i], nonlin));
    }
}

template <typename T, size_t N>
Value_Vec<T> MLP<T, N>::operator()(const Value_Vec<T> &x) {

    m_layers_output.push_back(std::make_shared<Value_Vec<T>>(x));

    for (size_t i = 1; i <= N; i++) {
        m_layers_output.emplace_back(std::make_shared<Value_Vec<T>>(
            m_layers[i - 1](*m_layers_output.back())));
    }

    // return the value of the last element which is a vector
    return *(m_layers_output.back());
}

template <typename T, size_t N>
Value<T> MLP<T, N>::MSE_loss_backprop(std::vector<Value_Vec<T>> &input,
                                      std::vector<Value_Vec<T>> &target, size_t batch_size) {

    Value_Vec<T> tmp_loss;
    std::vector<Value_Vec<T>> output;
    Value<T> loss = Value<T>(0.0, "loss");

    for (size_t i = 0; i < batch_size; i++) {
        // Forward pass - target
        output.emplace_back(operator()(input[i]));
    }

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < output[i].size(); j++) {
            tmp_loss.emplace_back(output[i][j] - target[i][j]);
        }
    }

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < output[i].size(); j++) {
            loss += tmp_loss[i] ^ 2.0;
        }
    }

    loss.backward();

    return loss;
}

template <typename T, size_t N>
std::vector<Value<T> *> MLP<T, N>::parameters() {
    std::vector<Value<T> *> params;
    // Iterate over all the layers
    for (auto &layer : m_layers) {
        auto layer_params = layer.parameters();
        // insert an object at the end thus we are not using emplace_back
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}
