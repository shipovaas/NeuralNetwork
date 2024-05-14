#pragma once

#include "../src/NeuralNetwork.h"

#include "../mnist/include/mnist/mnist_reader.hpp"

namespace neuralnet {
    //struct Data {
    //    Matrix input_vectors;
    //    Matrix output_vectors;
    //};
    //
    //struct Parameter {
    //    Matrix matrix_a;
    //    Vector vector_b;
    //};
    using data_type = double;
    using index = Eigen::Index;

    struct DataSet {
        index num_input_pixels;
        index num_output_pixels;
        index num_train_images;
        index num_test_images;
        Data train;
        Data test;
    };

    class MnistTesting {
    public:
        static void Run();

        static DataSet GetMnistData(Index train_size = kMnistTrainDataSize);

        static int Train(Net& net, DataSet& dataset, Index iter_count = kDefaultMaxIter,
                         DataType initial_learning_rate = kDefaultInitLR,
                         DataType decay = kDefaultDecay, LFName lf_name = kDefaultLFName,
                         const Path& path = "");

        static DataType CalcAccuracy(const Net& net, const DataSet& dataset);

        static constexpr const Index kMnistTrainDataSize = 60000;

    private:
        static constexpr const LFName kDefaultLFName = LFName::MSE;
        static constexpr const DataType kDefaultError = 0.01;
        static constexpr const Index kDefaultMaxIter = 25;
        static constexpr const DataType kDefaultInitLR = 0.1;
        static constexpr const DataType kDefaultDecay = 1;
        static constexpr const Index kDefaultBatchSize = 128;
        static constexpr const PI kDefaultPI = PI::PrintInfo;
    };
}  // namespace project