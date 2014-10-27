#ifndef DEVICE_MANAGER_HH
#define DEVICE_MANAGER_HH

#include <cuda.h>
#include <cuda_runtime.h>
#include <scarf/DataSet.hpp>
#include <scarf/cuda/cudaUtils.h>

namespace scarf {

    /**
     *  Class meant to manage all the memory operations between the device and
     *  the host.
     */
    class DeviceManager 
    {
    public:

        /** @brief Constructor.*/
        DeviceManager() {}

        /** @brief Destructor.*/
        ~DeviceManager();
        
        /**
         *  \brief Allocate the necessary memory in the device for the training
         *  data
         *
         *  @param nSamples Number of training examples.
         *  @param nFeatures Number of features for example.
         *  @param featureSize Size in bytes for each feature.
         */
        void trainingDataMemAlloc(
                size_t nSamples, 
                size_t nFeatures, 
                size_t featureSize);

        /**
         *  \brief Load the the training data into the device.
         *
         *  @param dt Dataset to construct the forest.
         */
        void moveDataToDevice(DataSet &dt);

        /** @brief Pointer to the data array in the device.*/
        float *d_data;

        /** @brief Pointer to the features array in the device.*/
        float *d_features;

        /** @brief Pointer to the labels array in the device.*/
        float *d_labels;

        /** @brief Pointer to the data array in the host.*/
        float *h_data;

        /** @brief Pitch size in the device.*/
        size_t pitch;

        /** @brief Total number of the samples.*/
        size_t ne;

        /** @brief Total number of the features.*/
        size_t nf;
    };
}

#endif
