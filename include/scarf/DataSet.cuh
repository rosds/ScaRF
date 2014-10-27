#ifndef DataSet_HH
#define DataSet_HH

#include <stdlib.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include <boost/shared_ptr.hpp>

#include <thrust/host_vector.h>

#include <scarf/Sample.cuh>

/**
 *  @file DataSet.hpp
 *
 *  @brief In this file is defined the class which contains the information
 *  corresponding to the data set employed for training the algorithm.
 */
namespace scarf {


    /**
     *  @brief Contains the information referring the training set.
     */
    template <class SampleT>
    class DataSet 
    {
        public:
            typedef boost::shared_ptr<DataSet<SampleT> > Ptr;


        /**
         *  @brief This class is designed to actually contain a subset of the
         *  entire data set to be employed in the training of the random forest.
         */
        class TrainingSet
        {
            public:
                /** @brief Pointer to the Training Set.*/
                typedef boost::shared_ptr<TrainingSet> Ptr;

            public:
                /** @brief Constructor.*/
                TrainingSet(size_t nSamples);

                /** @brief Destructor.*/
                ~TrainingSet() {}

                /** @brief Easy access to the training samples.*/
                SampleT& operator[](size_t idx);

                /** @brief Training data */
                thrust::host_vector<SampleT> trainingData;

                /** @brief Number of the samples in the training set */
                size_t size;
        };


        public:
            /** @brief Constructor.*/
            DataSet() : nSamples(0) {}

            /** @brief Destructor.*/
            ~DataSet() {}

            /** 
             *  @brief Get random subset of the training data to train each tree.
             *
             *  @param n Number of samples to get.
             */
            typename TrainingSet::Ptr subsample(const size_t n,
                    std::vector<size_t> *oobIdx);


            /**
             *  @brief Push back samples to the data set.
             *
             *  This function pushes a new sample to the vector containing the
             *  data samples.
             *
             *  @param sample The sample to be added in the vector.
             */
            void push_back(SampleT x)
            {
                data_samples.push_back(x);
                nSamples++;
            }


            /**
             *  @brief Load the data from CSV file.
             *
             *  TODO: Either delete this function or do it right.
             *
             *  At this moment this function is only for testing purposes.
             *
             *  @param filename CSV file name.
             *  @param ns Number of samples to read.
             *  @param nf Number of features per sample.
             */
            void loadCSVFile(std::string filename, size_t ns, size_t nf);

            /** @brief Operator to easy access on the samples */
            SampleT& operator[](size_t idx)
            {
                return data_samples[idx];
            }

            /** @brief Total Number of the Training Samples.*/
            size_t nSamples;

        private:
            /** @brief Vector of Training Samples.*/
            std::vector<SampleT> data_samples;
    };


    template <class SampleT>
    DataSet<SampleT>::TrainingSet::TrainingSet(size_t ns)
    {
        size = ns;
        trainingData = thrust::host_vector<SampleT>(ns);
    }


    template <class SampleT>
    SampleT& DataSet<SampleT>::TrainingSet::operator[](size_t idx)
    {
        return trainingData[idx];
    }


    template <class SampleT>
    typename scarf::DataSet<SampleT>::TrainingSet::Ptr 
    scarf::DataSet<SampleT>::subsample(const size_t n, std::vector<size_t> *oobIdx) {
        // Initialize random seed.
        srand (time(NULL));

        std::vector<size_t> randIdx;

        // Generate the random indices for the training set
        for (size_t i = 0; i < n; i++) { 
            randIdx.push_back(rand() % nSamples);
        }
        std::sort(randIdx.begin(), randIdx.end());

        // Create training set of size n
        boost::shared_ptr<TrainingSet> ts(new TrainingSet(n));

        // Fill the training set with the element at the random indices
        for (size_t i = 0; i < n; i++) {
            (*ts)[i] = data_samples[randIdx[i]];
        }

        // Fill the out of bag indices
        oobIdx->clear();
        for (size_t i = 0; i < nSamples; i++) {
            if (!std::binary_search(randIdx.begin(), randIdx.end(), i)) {
                oobIdx->push_back(i);
            }
        }

        return ts;
    }


    template <class SampleT>
    void scarf::DataSet<SampleT>::loadCSVFile(std::string filename, 
            size_t ns, size_t nf) {

        std::ifstream file(filename.c_str());
        std::string line;
        size_t count = 0;
        while(std::getline(file, line) && count < ns) {
            std::istringstream linestream(line);
            
            std::string label;
            SampleT t;

            // Read label
            std::getline(linestream, label, ',');
            std::istringstream labelStream(label);
            labelStream >> t.label;

            // Read Data
            std::string feature;
            size_t i = 0;
            while(std::getline(linestream, feature, ',')) {
                std::istringstream featureStream(feature);
                featureStream >> t.data[i];
                i++;
            }
            push_back(t);
            count++;
        }
        file.close();
    }
}   // namespace scarf


#endif
