#ifndef CUBEMAP_H
#define CUBEMAP_H
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "utils.h"

class CubeMap
{
public:
    CubeMap();
    CubeMap(unsigned int face_size);
    void loadCubeCross(const cv::Mat &cube_cross);
    std::vector<cv::Mat> exportIntoImages() const;
    cv::Mat exportCubeCross() const;
    void resizeCube(unsigned int new_face_size);
    void BluringFilter();
    unsigned int getFaceSize() const;

    //copy
    CubeMap& operator=(const CubeMap& other);
    //cube coords
    cv::Vec3f& operator [] (const cv::Vec3f& cube_coords);
    cv::Vec3f operator [] (const cv::Vec3f& cube_coords) const;
    //local face coords
    cv::Vec3f& operator () (const cv::Vec3f& face_coords);
    cv::Vec3f operator () (const cv::Vec3f& face_coords) const;
    //get spherical coords
    cv::Vec3f getSphericalCoords(const cv::Vec3f& face_coords) const;
    //get cubic coords
    cv::Vec3f getCubeCoords(const cv::Vec3f& face_coords) const;
    cv::Vec3f getCubeCoords(unsigned int face, const cv::Vec2f &face_coords) const;
    //get face x y coords from cubicx coords
    cv::Vec3f getFaceCoords(const cv::Vec3f& cube_coords) const;

private:
    std::vector<cv::Mat> cube;
    unsigned int cube_face_size;
    //int image_type;
};

#endif // CUBEMAP_H
