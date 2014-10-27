#define BOOST_NOINLINE __attribute__ ((noinline))


#include <iostream>
#include <fstream>
#include <stdlib.h>

#include <scarf/RF.cuh>


typedef size_t Phi;

/**
 *  This function generate different parameters to be used in each decision
 *  tree as part of the training.
 */
Phi generateParameter(size_t n) {
    return rand() % 2;
}


struct MySample : public scarf::Sample<int, float, 2> {

    __host__ __device__
    float toFeature(size_t idx) const {
        return data[idx];
    }
};




int main()
{
    // Initialize random seed.
    srand(time(NULL));

    scarf::RFConfig config;
    config.dataSetPath       = "../samples/data/samples.csv";
    config.nSamples          = 3000;
    config.nFeatures         = 2;
    config.nTrees            = 1;
    config.samplesPerTree    = 10;
    config.maxDepth          = 5;
    config.minSamplesPerNode = 1;
    config.numParameters     = 2;
    config.numThresholds     = 1;
    config.featureSize = sizeof(float);

    scarf::RF<Phi, MySample> jungle(config, generateParameter);
    jungle.train();

    std::ofstream predictionFile, confidenceFile;
    predictionFile.open("prediction.csv");
    confidenceFile.open("confidence.csv");
    for (float i = 0.0f; i <= 9.0f; i += 0.25f) {
        for (float j = -1.0f; j <= 5.0f; j += 0.25f) {
            MySample t;
            t.data[0] = j;
            t.data[1] = i;
            float confidence;
            int label = jungle.predict(t, &confidence);
            if (j == 5.0f) {
                predictionFile << label << "\n";
                confidenceFile << confidence << "\n";
            } else {
                predictionFile << label << ",";
                confidenceFile << confidence << ",";
            }
        }
    }
    predictionFile.close();
    confidenceFile.close();

    return 0;
}
