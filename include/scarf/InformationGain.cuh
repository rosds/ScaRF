#ifndef SCARF_INFORMATIONGAIN_HH
#define SCARF_INFORMATIONGAIN_HH

#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>


namespace scarf {


/**
 *  @brief Functor to obtain the probability density of each label.
 *
 *  This functor is used in the Shannon entropy calculation. The result is later
 *  reduced in a sum to get the entire entropy.
 */
struct p : public thrust::unary_function<int, float>
{
    size_t size;
    p(size_t _s) : size(_s) {}
    __device__
    float operator()(const int x) {
        if (x == 0) {
            return 0.0f;
        } else {
            float prob = (float) x / (float) size;
            return - prob * log(prob);
        }
    }
};


/**
 *  @brief Calculate the Shannon entropy of a histogram.
 *
 *  This function calculates the Shannon Entropy of a histogram. The histogram
 *  should be a device_vector of type int, which every position of the vector
 *  represents a label and contains the total amount of labels of that type.
 *
 *  @param hist A thrust's device vector containing the histogram of classes.
 *  @returns Shannon entropy of the histogram.
 *
 */
float H(thrust::device_vector<int>& hist)
{
    size_t n = thrust::reduce(hist.begin(), hist.end(), 0, thrust::plus<int>());

    thrust::device_vector<float> prob(hist.size());
    thrust::transform(hist.begin(), hist.end(), prob.begin(), p(n));
    return thrust::reduce(prob.begin(), prob.end(), 0.0f, thrust::plus<float>());
}


/**
 *  @brief Calculates the information gain in the split of the data.
 *
 *  This functions receives two histograms of the label occurrences on left and
 *  right split. The information gain is calculated by subtracting the resulting
 *  entropy of the split to the entropy of the entire set.
 *
 *  @param left Left histogram for the label occurrences. 
 *  @param right Right histogram for the label occurrences. 
 *  @param entireSetEntropy Pre-computed entropy of the entire set.
 *  @param setSize Size of the entire set.
 *  @returns Information gain of the particular split.
 */
float informationGain(
        thrust::device_vector<int>& left, 
        thrust::device_vector<int>& right,
        const float entireSetEntropy,
        const float setSize
)
{
    float rSize;
    float lSize;

    if (setSize == 0) return 0.0f;

    // Get the size of each set
    rSize = thrust::reduce(right.begin(), right.end(), 0.0f, 
            thrust::plus<float>());
    lSize = thrust::reduce(left.begin(), left.end(), 0.0f, 
            thrust::plus<float>());

    return entireSetEntropy - (rSize * H(right) + lSize * H(left)) / setSize;
}


}   // namespace scarf


#endif // SCARF_INFORMATIONGAIN_HH
