#ifndef USER_DEFINITIONS_HH
#define USER_DEFINITIONS_HH


#include <vector>


namespace scarf {


template<class LabelT, class FeatureT, size_t N>
struct Sample {
    public:
        // Make types accessible from outsize
        typedef LabelT LabelType;
        typedef FeatureT FeatureType;

    public:
        /** @brief Label of the sample */
        LabelType label;

        /** @brief Data vector for the samples */
        FeatureType data[N];

        __host__ __device__
        FeatureType& operator[](size_t idx) {return data[idx];}

        __host__ __device__
        const FeatureType& operator[](size_t idx) const {return data[idx];}

        __host__ __device__
        size_t size() {return N;}
        
        // TODO: this is nasty
        __host__ __device__
        operator int() const {return label;}

        __host__ __device__
        friend std::ostream &operator<<(std::ostream &os, Sample const &t) {
            os << t.label << ", " << t.data[0] << ", " << t.data[1] << std::endl;
            return os;
        }

        /*
         *__host__ __device__
         *virtual FeatureType toFeature(size_t i) const {
         *    return data[i];
         *}
         */
};

}   // namespace scarf

#endif // USER_DEFINITIONS_HH
