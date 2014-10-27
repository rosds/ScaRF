#ifndef SCARF_TREE_TREE_HH
#define SCARF_TREE_TREE_HH

// Standard libraries
#include <vector>
#include <iostream>

// Third-party libraries
#include <boost/shared_ptr.hpp>

#include <thrust/device_vector.h>
#include <thrust/sort.h>

// Scarf headers
#include <scarf/Tree/TreeTrainingUtils.cuh>


/**
 *  @file Tree.cuh
 *
 *  @brief In this file we define the Tree structure together with the necessary
 *  classes like SplitNode and LeafNode.
 *
 */
namespace scarf {


/**
 *  @brief Decision Tree
 *
 *  This is a template tree that in each node it contains the parameters
 *  specified by the user and a threshold. Also each node extract the feature
 *  to threshold from the data using the parameters.
 */
template <class SplitParam, class SampleT>
class Tree {
    public:

        typedef SplitParam (*ParamSamplingFun)(size_t);
        typedef boost::shared_ptr<Tree<SplitParam, SampleT> > Ptr;
        typedef thrust::device_vector<SampleT> TrainingSetVector;
        typedef typename thrust::device_vector<SampleT>::iterator TrainingSetIter;

        /**
         *  @brief Internal node of the decision tree.
         *
         *  This node emits a binary prediction of the data. On the training it
         *  is created with the parameters and threshold necessary to separate
         *  the data into two.
         */
        class Node {
            public:
                
                /** @brief Empty constructor. */
                Node() {}

                /** 
                 * @brief Constructor.
                 *
                 * @param p Parameter for feature extraction function.
                 * @param t Threshold used in the feature extracted to split
                 * the data.
                 */
                Node(SplitParam p, float t) : params(p), threshold(t) {}

                /** @brief Destructor. */
                ~Node() {}

                /** 
                 *  @brief Prediction function.
                 *
                 *  Extracts the feature from the data point using the internal
                 *  parameters and then thresholds it to return the
                 *  corresponding chield node's index.
                 *
                 *  @param x Data sample to classify.
                 *  @returns Chind node index to continue the prediction.
                 */
                int predict(const SampleT& x);

                /** @brief Left child index. */
                int leftChildId;

                /** @brief Right child index. */
                int rightChildId;

            private:

                /** @brief Internal parameters for feature extraction. */
                SplitParam params;

                /** @brief Threshold for the feature. */
                float threshold;
        };


        /**
         *  @brief Leaf Node containing the class labels probability
         *  distribution.
         *
         *  This basic structure only contains the probability density of the
         *  class labels obtained during training. Later on these distributions
         *  are used to the final classification.
         */
        class ClassDensity {
            public:

                /** 
                 *  @brief Constructor.
                 *
                 *  This constructor takes two iterators on a certain segment
                 *  of the training set and the build a corresponding
                 *  probability distribution based on the occurrences of the
                 *  labels in that segment. 
                 *  
                 *  @param start Iterator to the beginning of the segment from
                 *  where to build the probability distribution.
                 *  @param end Iterator to the end of the segment from
                 *  where to build the probability distribution.
                 */
                ClassDensity(TrainingSetIter start, TrainingSetIter end);

                /** @brief Destructor. */
                ~ClassDensity() {} 

                thrust::host_vector<float> classDensity;

                thrust::host_vector<float> classLabels;
        };

    public:

        /**
         *  @brief Constructor.
         *
         *  @param trainSet Training set pointer to use for learning.
         */
        Tree (typename scarf::DataSet<SampleT>::TrainingSet::Ptr _trainSet, 
                size_t _maxDepth, size_t _minSamplesPerNode, size_t _numThesholds, size_t _numParameters) {
            trainingSet = _trainSet; 
            maxDepth = _maxDepth;
            minSamplesPerNode = _minSamplesPerNode;
            numThresholds = _numThesholds; 
            numParameters = _numParameters;
        }

        /** @brief Destructor. */
        ~Tree () {}

        /**
         *  @brief Training method
         *
         *  Uses the training set to grow a tree.
         */
        void train();

        /**
         *  @brief predict.
         */
        int predict(const SampleT& x);

        /**
         *  @brief predict.
         */
        void predict(const SampleT& x, thrust::host_vector<float>
                *probabilities, thrust::host_vector<float> *lables);


        /**
         *  @brief Set the generation function for split parameters.
         *
         *  This function depends on the goal for which the tree is
         *  designed. It should be provided by the user or application using
         *  the tree.
         *
         *  This function is used to generate the split parameters that are
         *  going to be used as candidates for the creation of each
         *  splitting node of the tree.
         *
         *  @param func Function that returns the parameters which will be
         *  later given to the functor function.
         */
        void setParameterGenFunction(ParamSamplingFun func);

    private:

        /** @brief Data for training.*/
        typename DataSet<SampleT>::TrainingSet::Ptr trainingSet;

        /** @brief Device vector to contain the data set */
        TrainingSetVector deviceTrainingSetVector;

        /** @brief Vector containing the inner nodes of the tree */
        std::vector<Node> treeNodes;

        /** @brief Vector containing the leaf nodes of the tree. */
        std::vector<ClassDensity> leafNodes;

        /** @brief Root node index */
        int rootIdx;

        /** @brief Maximum Depth in the Tree. */
        size_t maxDepth;

        /** @brief Minimum number of samples per node. */
        size_t minSamplesPerNode;

        /** @brief Number of thresholds to consider. */
        size_t numThresholds;

        /** @brief Number of parameters to generate. */
        size_t numParameters;

        /**
         *  @brief Pointer to parameter generation function.
         *
         *  This function is defined by the user and depends on the desired
         *  function of the tree.
         */
        ParamSamplingFun sampleWeakLearner;

        /**
         *  @brief Depth-first tree training fashion.
         */
        int depthFirstTraining(
            const size_t depth, 
            TrainingSetIter start, 
            TrainingSetIter end);

        /**
         *  @brief Create a leaf node with the class label densities of the
         *  specified training set segment.
         *
         *  @param start Iterator pointing to the beginning of the segment.
         *  @param end Iterator pointing to the end of the segment.
         *  @returns Negative index to the array of leafNodes.
         */
        int createLeafNode(TrainingSetIter start, TrainingSetIter end);
};


template <class SplitParam, class SampleT>
scarf::Tree<SplitParam, SampleT>::ClassDensity::ClassDensity(
        TrainingSetIter start, TrainingSetIter end) {

    thrust::device_vector<float> density, labels;
    getLabelDensity<SampleT>(start, end, density, labels);
    classDensity = density;
    classLabels = labels;
}


template <class SplitParam, class SampleT>
int scarf::Tree<SplitParam, SampleT>::Node::predict(const SampleT& x) {
    float feature = x.toFeature(params);
    if (feature < threshold) {
        return leftChildId;
    } else {
        return rightChildId;
    }
}


template <class SplitParam, class SampleT>
void scarf::Tree<SplitParam, SampleT>::train() {
    deviceTrainingSetVector = trainingSet -> trainingData;
    TrainingSetIter start = deviceTrainingSetVector.begin();
    TrainingSetIter end = deviceTrainingSetVector.end();

    rootIdx = depthFirstTraining(0, start, end);
}

template <class SplitParam, class SampleT>
int scarf::Tree<SplitParam, SampleT>::predict(const SampleT& x) {

    Node node;
    int nodeIdx = rootIdx;
    while (nodeIdx >= 0) {
        node = treeNodes[nodeIdx];
        nodeIdx = node.predict(x);
    }
    ClassDensity leaf = leafNodes[abs(nodeIdx) - 1];

    thrust::host_vector<float>::iterator maxElem;
    maxElem = thrust::max_element(leaf.classDensity.begin(), leaf.classDensity.end());
    size_t labelIdx = maxElem - leaf.classDensity.begin();

    return leaf.classLabels[labelIdx];
}

template <class SplitParam, class SampleT>
void scarf::Tree<SplitParam, SampleT>::predict(const SampleT& x,
        thrust::host_vector<float> *probabilities, thrust::host_vector<float>
        *labels) {

    Node node;
    int nodeIdx = rootIdx;
    while (nodeIdx >= 0) {
        node = treeNodes[nodeIdx];
        nodeIdx = node.predict(x);
    }
    ClassDensity leaf = leafNodes[abs(nodeIdx) - 1];
    probabilities->resize(leaf.classDensity.size());
    labels->resize(leaf.classLabels.size());

    thrust::copy(leaf.classDensity.begin(), leaf.classDensity.end(),
            probabilities->begin());
    thrust::copy(leaf.classLabels.begin(), leaf.classLabels.end(),
            labels->begin());
}


template <class SplitParam, class SampleT>
void scarf::Tree<SplitParam, SampleT>::setParameterGenFunction(
    ParamSamplingFun fun) {

    this -> sampleWeakLearner = fun;
}


template <class SplitParam, class SampleT>
int scarf::Tree<SplitParam, SampleT>::createLeafNode(
        TrainingSetIter start, TrainingSetIter end) {
        ClassDensity leaf(start, end);
        leafNodes.push_back(leaf);
        return -1 * ((int)leafNodes.size());
}


template <class SplitParam, class SampleT>
int scarf::Tree<SplitParam, SampleT>::depthFirstTraining(
    size_t depth, TrainingSetIter start, TrainingSetIter end) {


    // Max depth reach, build the leaf and return.
    if (depth > maxDepth) {
        return createLeafNode(start, end);
    }

    size_t segmentSize;
    if (start < end) {
        segmentSize = end - start;
    } else {
        std::cout << "WARNING: Hey this should not happen" << std::endl;
        segmentSize = 0; 
    }

    if (segmentSize <= minSamplesPerNode) {
        return createLeafNode(start, end);
    }

    thrust::device_vector<float> featureVector(segmentSize);
    thrust::device_vector<int> labelVector(segmentSize);

    // Set label vector
    setLabelVector(start, end, labelVector);
    
    // Check when the number of labels are uniform
    size_t numLabels = countUniqueValues(labelVector);
    if (numLabels == 1) {
        return createLeafNode(start, end);
    }

    float bestInfoGain = -1.0f;
    float bestThreshold;
    SplitParam bestParameters;
    size_t sPoint;
    bool split = false;

    for (size_t p = 0; p < numParameters; p++) {

        // Sample a weak learner
        SplitParam phi = sampleWeakLearner(p);

        print_vector("Label   vector", labelVector);
        // Set feature vector
        setFeatureVector(phi, start, end, featureVector);
        print_vector("Label   vector", featureVector);

        // Set label vector
        setLabelVector(start, end, labelVector);

        // Sort by feature
        thrust::sort_by_key(featureVector.begin(), featureVector.end(),
                labelVector.begin());

        // Find threshold
        float threshold;
        float infoGain;
        size_t splitPoint;

        findBestThreshold(1, featureVector, labelVector, &threshold, &infoGain,
                &splitPoint);
        
        if (infoGain > bestInfoGain) {
            bestInfoGain = infoGain;
            bestThreshold = threshold;
            bestParameters = phi;
            sPoint = splitPoint;
            split = true;
        }
    }
    
    if (!split) {
        return createLeafNode(start, end);
    }

    std::cout << "Creating split node with:" << std::endl;
    std::cout << " Threhold: " << bestThreshold << std::endl;
    std::cout << " InfoGain: " << bestInfoGain << std::endl;
    std::cout << " Feature:  " << bestParameters << std::endl;
    std::cout << std::endl;

    // Sort the training set with the selected feature
    setFeatureVector(bestParameters, start, end, featureVector);
    thrust::sort_by_key(featureVector.begin(), featureVector.end(), start);

    Node newNode(bestParameters, bestThreshold);
    newNode.leftChildId = depthFirstTraining(depth + 1, start, start + sPoint);
    newNode.rightChildId = depthFirstTraining(depth + 1, start + sPoint, end);

    int myIdx = treeNodes.size();
    treeNodes.push_back(newNode);

    // Return idx of node in the array
    return myIdx;
}


}   // namespace scarf


#endif
