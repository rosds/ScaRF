#ifndef SCARF_RF_HH
#define SCARF_RF_HH

// Std includes
#include <ctime>
#include <map>
#include <vector>

// Third-party libraries
#include <boost/shared_ptr.hpp>

// Scarf Includes
#include <scarf/RFConfig.cuh>
#include <scarf/DataSet.cuh>
#include <scarf/Tree/Tree.cuh>

namespace scarf {

    template <class SplitParam, class SampleT>
    class RF 
    {
        public:
            typedef SplitParam (*SampleParamFun)(size_t);
            typedef boost::shared_ptr<RF<SplitParam, SampleT> > Ptr;


        public:
            /**
             * @brief Constructor.
             * @param conf Configuration structure for the Random Forest.
             */
            RF(RFConfig conf, SampleParamFun genParam);

            /** @brief Destructor. */
            ~RF() {}

            /** @brief Set random forest data set */
            void setDataSet(typename DataSet<SampleT>::Ptr dataSet);

            /** @brief Train the Random Forest. */
            void train();

            /**
             *  @brief Predict the label of the sample.
             *
             *  Returns the label with higher number of votes by all the trees.
             *
             *  @param x Sample to predict the label
             *  @returns The predicted label.
             */
            float predict(const SampleT &x, float *confidence);

            float quickCheck();

        private:

            /** @brief Random Forest Configuration class */
            RFConfig conf;

            /** @brief Data set for training. */
            typename DataSet<SampleT>::Ptr data;

            /** @brief Array of decision trees */
            std::vector<typename Tree<SplitParam, SampleT>::Ptr> forest;

            /**
             *  @brief Pointer to parameter generation function.
             *
             *  This function is defined by the user and depends on the desired
             *  function of the tree.
             */
            SampleParamFun genParam;
    };
}



template <class SplitParam, class SampleT>
scarf::RF<SplitParam, SampleT>::RF (
        scarf::RFConfig config,
        SampleParamFun genFun) : data(new DataSet<SampleT>()) {
    conf = config;
    genParam = genFun;
    data -> loadCSVFile(config.dataSetPath, config.nSamples, config.nFeatures);
}


template <class SplitParam, class SampleT>
void scarf::RF<SplitParam, SampleT>::train() {

    size_t n = conf.nTrees; // Number of trees to train.
    size_t subSetSize = conf.samplesPerTree; // Size of training sets for trees.
    size_t maxDepth = conf.maxDepth; // Maximum depth of a tree.
    size_t minSamplesPerNode = conf.minSamplesPerNode;
    size_t nThresholds = conf.numThresholds;
    size_t nParameters = conf.numParameters;
    std::vector<size_t> oobIdx;
    std::vector<std::map<int, int> > oobHistograms(data->nSamples);

    clock_t begin = std::clock();

    // Main for loop in the algorithm: Train each tree separately.
    for (size_t i = 0; i < n; i++) {
        std::cout << "Training Tree " << i << "...";
        std::cout.flush();

        // Subsample a set from DataSet
        typename DataSet<SampleT>::TrainingSet::Ptr subset = data ->
            subsample(subSetSize, &oobIdx);

        // Create tree from the subset of samples.
        typename Tree<SplitParam, SampleT>::Ptr tree(
                new Tree<SplitParam, SampleT>(subset, maxDepth,
                    minSamplesPerNode, nThresholds, nParameters)); 

        // Set tree parameter generation function
        tree -> setParameterGenFunction(genParam);

        // Train the tree.
        tree -> train();

        // Get the oob classification
        for (size_t j = 0; j < oobIdx.size(); j++) {
            int label;
            SampleT oobSample = (*data)[oobIdx[j]];
            label = tree->predict(oobSample);
            if (label != oobSample.label) {
                oobHistograms[oobIdx[j]][label] += 1;
            }
        }

        // Add the trained tree to the forest.
        forest.push_back(tree); 
        std::cout << "done" << std::endl;
    }

    clock_t end = std::clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "Training Time: " << elapsed_secs << std::endl;

    // Calculate the oob error
    float error = 0.0f;
    for (size_t i = 0; i < data->nSamples; i++) {
        std::map<int, int>::iterator iter;
        size_t count = 0;
        int max = 0;
        for (iter = oobHistograms[i].begin(); iter != oobHistograms[i].end();
                iter++) {
            if (iter->second > max) {
                max = iter->second;
            }
            count += iter->second;
        } 
        error += count > 0 ? (float)max / (float)count : 0.0f;
    }
    std::cout << "OOB error: " << error / (float)data->nSamples << std::endl;
}


template <class SplitParam, class SampleT>
void scarf::RF<SplitParam, SampleT>::setDataSet(
        typename DataSet<SampleT>::Ptr dataSet) {
    data = dataSet;
}


template <class SplitParam, class SampleT>
float scarf::RF<SplitParam, SampleT>::predict(const SampleT& x, float *confidence) {

    thrust::host_vector<float> probabilities, labels;

    std::map<float, float> labelsProbSum;

    for (size_t i = 0; i < conf.nTrees; i++) {
        forest[i]->predict(x, &probabilities, &labels); 

        // Sum the probabilities
        for (size_t j = 0; j < labels.size(); j++) {
            labelsProbSum[labels[j]] += probabilities[j];
        }
    }

    probabilities.clear();
    labels.clear();

    for (std::map<float, float>::iterator i = labelsProbSum.begin(); 
            i != labelsProbSum.end(); i++) {
        labels.push_back(i->first);
        probabilities.push_back(i->second / (float)conf.nTrees);
    }

    thrust::host_vector<float>::iterator maxElem;
    maxElem = thrust::max_element(probabilities.begin(), probabilities.end());

    (*confidence) = *maxElem;

    return labels[maxElem - probabilities.begin()];
}


template <class SplitParam, class SampleT>
float scarf::RF<SplitParam, SampleT>::quickCheck() {
    size_t s;
    float error = 0.0f;
    for (s = 0; s < data -> nSamples; s++) {
        float confidence;
        int label = predict((*data)[s], &confidence);
        if (label != (*data)[s].label) {
            error += 1.0f;
        }
    }
    error /= (float)data -> nSamples;
    std::cout << "Error " << error << std::endl;
    return error;
}

#endif  // define SCARF_RF_HH
