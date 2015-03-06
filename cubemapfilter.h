#ifndef CUBEMAPFILTER_H
#define CUBEMAPFILTER_H
#include <cmath>
#include <omp.h>
#include "cubemap.h"
#include "utils.h"

class CubeMapFilter
{
public:
    CubeMapFilter();

    static CubeMap cosineFilterCube(CubeMap& src, float angle);
    static CubeMap cosinePowFilterCube(const CubeMap& src, float angle, float pow);
    static std::vector<CubeMap> cosinePowFilterArrayCube(CubeMap& src, float angle, const std::vector<float> &pows);
    static CubeMap edgePullFixup(CubeMap &src, unsigned int len_fixup_band = 1);
    static CubeMap edgeSmoothFixup(CubeMap &src, unsigned int len_fixup_band = 1);


private:
    template <typename T> int sgn(T val) {
        return (T(0) < val) - (val < T(0));
    }
    static std::vector<std::vector<std::vector<cv::Vec3f> > > fillSphericalCoordsTable(const CubeMap& src);
    static float pixelIntensity(const cv::Vec3f &vec);
    static const unsigned int oposite_face[6];
    static CubeMap edgeFixup(CubeMap src, unsigned int len_fixup_band, bool pull);
};

#endif // CUBEMAPFILTER_H
