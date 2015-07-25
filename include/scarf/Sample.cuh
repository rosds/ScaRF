#ifndef USER_DEFINITIONS_HH
#define USER_DEFINITIONS_HH


#include <vector>


namespace scarf {


/** \brief Represents a feature vector with its corresponding label.
 *  
 *  This class holds a feature vector from which the random forest algorithms 
 *  tries to learn to classify. Associated to this feature vector, is the label 
 *  which the algorithm is suppose to learn.
 *
 *  \tparam LabelT Type of the label associated to the data.
 *  \tparam FeatureT Type of the data contained by the vector.
 *  \tparam N The size of the feature vector.
 */
template<class LabelT, class FeatureT, size_t N>
class Sample {
    public:
        typedef LabelT LabelType;
        typedef FeatureT FeatureType;

    public:
        /** \brief Label of the sample */
        LabelType label;

        /** \brief Data vector for the samples. */
        FeatureType data[N];

        /** \brief Operator to access the feature vector. */
        __host__ __device__
        FeatureType& operator[](size_t idx) {return data[idx];}

        /** \brief Operator to access the feature vector by reference. */
        __host__ __device__
        const FeatureType& operator[](size_t idx) const {return data[idx];}

        /** \brief Returns the size of the feature vector. */
        __host__ __device__
        size_t size() {return N;}
        
        // TODO: this is nasty
        __host__ __device__
        operator int() const {return label;}

        // TODO: Remove this
        __host__ __device__
        friend std::ostream &operator<<(std::ostream &os, Sample const &t) {
            os << t.label << ", " << t.data[0] << ", " << t.data[1] << std::endl;
            return os;
        }
};

}   // namespace scarf

#endif // USER_DEFINITIONS_HH
