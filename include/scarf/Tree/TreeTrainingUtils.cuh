#ifndef SCARF_TREE_TREETRAININGUTILS_HH
#define SCARF_TREE_TREETRAININGUTILS_HH

#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>

// Third-party headers
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/unique.h>
#include <thrust/count.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>

#include <scarf/DataSet.cuh>
#include <scarf/InformationGain.cuh>


/**
 *  @brief TreeTrainingUtils.cuh
 *
 *  In this file we define miscellaneous functions  for the Training of a
 *  decision tree.
 *
 */
namespace scarf {



template <typename Vector>
void print_vector(const std::string& name, const Vector& v)
{
    typedef typename Vector::value_type T;
    std::cout << " " << std::setw(20) << name << " ";
    thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
    std::cout << std::endl;
}


/**
 *  @brief Functor to transform the data on  the device to feature space.
 */
template <class SplitParam, class SampleT>
struct CalculateFeature {
    SplitParam phi;

    CalculateFeature(SplitParam _phi) : phi(_phi) {}

    __host__ __device__
    float operator()(const SampleT& x) {
        return x.toFeature(phi);
    }
};


template <class SampleT>
struct ExtractLabel {
    __host__ __device__
    int operator()(const SampleT& x) {
        return x.label;
    }
};


template <typename Phi, typename RandomAccessIterator>
void setFeatureVector(
        const Phi phi,
        RandomAccessIterator start,
        RandomAccessIterator end,
        thrust::device_vector<float> &featureVector) {

    typedef typename RandomAccessIterator::value_type SampleT;
    thrust::transform(start, end, featureVector.begin(),
            CalculateFeature<Phi, SampleT>(phi));
}


template <typename RandomAccessIterator>
void setLabelVector(
        RandomAccessIterator start,
        RandomAccessIterator end,
        thrust::device_vector<int> &labelVector) {

    typedef typename RandomAccessIterator::value_type SampleT;
    thrust::transform(start, end, labelVector.begin(),
            ExtractLabel<SampleT>());
}


/**
 *  @brief Count the number of unique elements in a vector in parallel.
 *
 *  This function count the number of different value inside the given array by
 *  performing a sort operation in a copy of the vector and later zip the
 *  sorted vector with it self. The element in the position i is compared with
 *  the element in the position i + 1. The comparisons of different values will
 *  sum 1 to the total number of different values.
 *
 *  @param vector The input vector from which the different elements will be
 *  count.
 *  @returns Number of different elements in the input vector.
 */
template <typename DeviceVector>
size_t countUniqueValues(const DeviceVector &vector) {

    typedef typename DeviceVector::value_type ValueType;

    if (vector.size() == 0) return 0;

    DeviceVector vec(vector);
    thrust::sort(vec.begin(), vec.end());

    size_t num = inner_product(
            vec.begin(), vec.end() - 1,
            vec.begin() + 1,
            size_t(1),
            thrust::plus<ValueType>(),
            thrust::not_equal_to<ValueType>()
    );

    return num;
}


/**
 *  @brief Gets a sorted array with the elements inside the input
 *  vector without repetition.
 *
 *  The input vector is sorted and then the unique elements are searched. The
 *  output is a vector containing the unique elements of the input array.
 *
 *  @param vector Input vector.
 *  @returns A sorted vector with the elements of the input vector but without
 *  repetition.
 */
template <typename DeviceVector>
DeviceVector uniqueElements(const DeviceVector &vector) {

    typedef typename DeviceVector::iterator Iterator;

    if (vector.size() == 0) return DeviceVector(0);

    DeviceVector vec(vector);
    thrust::sort(vec.begin(), vec.end());
    Iterator end = thrust::unique(vec.begin(), vec.end());

    return DeviceVector(vec.begin(), end);
}


template <class SampleT, typename RandomAccessIterator>
void getLabelDensity(
        RandomAccessIterator start, RandomAccessIterator end,
        thrust::device_vector<float>& density,
        thrust::device_vector<float>& label) {

    if (start > end) return;

    const size_t n = end - start;

    thrust::device_vector<float> labels(n);
    thrust::transform(start, end, labels.begin(), ExtractLabel<SampleT>());
    thrust::sort(labels.begin(), labels.end());

    // number of histogram bins is equal to number of unique values.
    int num_bins = countUniqueValues(labels);

    density.resize(num_bins);
    label.resize(num_bins);

    thrust::reduce_by_key(
            labels.begin(), labels.end(),
            thrust::constant_iterator<float>(1.0f),
            label.begin(),
            density.begin(),
            thrust::equal_to<float>());

    thrust::device_vector<float> div(n);
    thrust::fill(div.begin(), div.end(), (float)n);
    thrust::transform(density.begin(), density.end(), div.begin(),
            density.begin(), thrust::divides<float>());
}


/**
 *  @brief Calculate the k + 1 histograms
 */
template <typename FeatureVector, typename LabelVector>
void findBestThreshold(
        const size_t numThresholds, 
        const FeatureVector &featureVec,
        const LabelVector &labelVec,
        float *threshold,
        float *info,
        size_t *splitPosition) {

    // Sample type
    typedef typename thrust::device_vector<int>::iterator HistogramIterator;

    // TODO: Checks we may do here
    // if data.size() < numThresholds?
    // if data.size() == 1? == 0?

    const size_t n = featureVec.size();
    const size_t numHist = numThresholds + 1;


    LabelVector labels(labelVec);

    // Initialize histograms
    thrust::device_vector<int> lHist = uniqueElements(labelVec);
    thrust::device_vector<int> cHist[numHist];

    // Compute thresholds
    float min = featureVec[0];
    float max = featureVec[n - 1];
    float thresholdInc = (max - min) / (numThresholds + 1);

    thrust::device_vector<float> thresholds(numThresholds, 0.0f);
    size_t splitPositions[numThresholds];

    for (size_t i = 0; i < numThresholds; i++) {
        thresholds[i] = (i + 1) * thresholdInc + min;
    }

    // number of histogram bins is equal to number of unique values.
    int num_bins = countUniqueValues(labelVec);

    // Fill the histograms
    HistogramIterator start, end;
    start = labels.begin();
    for (size_t i = 0; i < numHist; i++) {

        // resize histogram storage
        cHist[i].resize(num_bins);

        // Set range of elements in the histogram
        if (i < numHist - 1) {
            float currentThres = thresholds[i];
            size_t pos = thrust::upper_bound(featureVec.begin(), featureVec.end(),
                    currentThres) - featureVec.begin();
            splitPositions[i] = pos;
            end = labels.begin() + pos;
        }
        if (i == numHist - 1) {
            end = labels.end();
        }

        for (size_t j = 0; j < num_bins; j++) {
            cHist[i][j] = thrust::count(start, end, lHist[j]);
        }

        start = end;
    }

    thrust::device_vector<int> left(num_bins, 0);
    thrust::device_vector<int> right(num_bins, 0);

    // Sum all the histograms
    for (size_t i = 0; i < numHist; i++) {
        thrust::transform(
                cHist[i].begin(), cHist[i].end(), 
                right.begin(), right.begin(),
                thrust::plus<int>());
    }

    float setEntropy = scarf::H(right);

    // Find best threshold
    size_t thresholdIdx = 0;
    float bestThreshold;
    float bestInfoGain = -1.0f;

    for (size_t t = 0; t < numThresholds; t++) {

        // Adds a histogram to the left accumulated one.
        thrust::transform(
                cHist[t].begin(), cHist[t].end(), 
                left.begin(), 
                left.begin(),
                thrust::plus<int>());

        // Subtract a histogram from the right accumulated one.
        thrust::transform(
                right.begin(), right.end(),
                cHist[t].begin(),
                right.begin(),
                thrust::minus<int>());
    
        float infoGain = informationGain(left, right, setEntropy, n);

        if (infoGain > bestInfoGain) {
            bestInfoGain = infoGain;
            bestThreshold = thresholds[t];
            thresholdIdx = t;
        }
    }

    // Return parameters
    *threshold = bestThreshold;
    *info = bestInfoGain;
    *splitPosition = splitPositions[thresholdIdx];
}


} // namespace scarf


#endif
