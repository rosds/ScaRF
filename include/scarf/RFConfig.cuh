#ifndef RFCONFIG_HH
#define RFCONFIG_HH

#include <string>

/**
 *  @file RFConfig.hpp
 *
 *  @brief In this file, the configuration of random forest is defined.
 */
namespace scarf {

    /** @brief Enumerater for data types.*/
    enum dType {
        RF_FLOAT,
        RF_INT 
    };

    /** @brief Random Forest Configuration Class. This class contains the
     *  configuration parameters of the random forest.
     */
    class RFConfig 
    {
    public:

        /** @brief Constructor.*/
        RFConfig() {}

        /** @brief Destructor.*/
        ~RFConfig() {}

        /** @brief Total number of the trees in the forest. */
        size_t nTrees;
    
        /** @brief Total number of the training samples. */
        size_t nSamples;

        /** @brief Total number of the features per training sample. */
        size_t nFeatures;

        /** @brief Size of the features in bytes. */
        size_t featureSize;

        /** @brief Number of samples per tree */
        size_t samplesPerTree;

        /** @brief Minimum number of samples per node */
        size_t minSamplesPerNode;

        /** @brief Maximum depth of the trees */
        size_t maxDepth;

        /** @brief Number of split candidates to try per node */
        size_t numParameters;

        /** @brief Number of threshold per parameter */
        size_t numThresholds;

        /** @brief Path to the data set directory */
        std::string dataSetPath;
    };
}

#endif
