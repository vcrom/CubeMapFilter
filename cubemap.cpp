#include "cubemap.h"
#include <opencv2/imgproc/imgproc.hpp>

CubeMap::CubeMap()
{
    cube = std::vector<cv::Mat> (6);
    cube_face_size = 0;
    //image_type = 0;
}

CubeMap::CubeMap(unsigned int face_size)
{
    cube = std::vector<cv::Mat> (6);
    for(unsigned int i = 0; i < 6; ++i)
    {
        cube[i] = cv::Mat(face_size, face_size, CV_32FC3, cv::Scalar(0));
    }
    cube_face_size = face_size;
}

const unsigned int cubes_w[6] = {2, 0, 1, 1, 1, 1};
const unsigned int cubes_h[6] = {1, 1, 0, 2, 1, 3};

void CubeMap::loadCubeCross(const cv::Mat &cube_cross)
{
    //cube = std::vector<cv::Mat> (6);
    cv::Size size = cube_cross.size();
    int im_width = size.width;
    cube_face_size = im_width/3;
    //image_type = cube_cross.type();
    for(unsigned int i = 0; i < 6; ++i)
    {
        cv::Rect rec(cube_face_size*cubes_w[i], cube_face_size*cubes_h[i], cube_face_size, cube_face_size);
//        cube[i] = cube_cross(rec);
        cube_cross(rec).copyTo(cube[i]);
//        cv::imshow("face", cube[i]);
//        cv::waitKey();
    }
//    cv::imshow("face", cube[5]);
//    cv::waitKey();
    cv::flip(cube[5], cube[5], 0);
//    cv::flip(cube[5], cube[5], -1);
//    cv::imshow("face", cube[5]);
//    cv::waitKey();
//    cv::flip(cube[2], cube[2], -1);

}

//    const char* texture_names[6] = {"enviroment_maps/ocean/posx.dds",
//                                    "enviroment_maps/ocean/negx.dds",
//                                    "enviroment_maps/ocean/posy.dds",
//                                    "enviroment_maps/ocean/negy.dds",
//                                    "enviroment_maps/ocean/posz.dds",
//                                    "enviroment_maps/ocean/negz.dds"};

CubeMap& CubeMap::operator=( const CubeMap& other )
{
    std::cout << "COPY" << std::endl;
//    cv::Mat orig_cross = other.exportCubeCross();
    this->cube = other.exportIntoImages();
    this->cube_face_size = other.getFaceSize();
//    this->loadCubeCross(orig_cross);
    return *this;
}

cv::Vec3f& CubeMap::operator [] (const cv::Vec3f& cube_coords)
{
    assert(cube_coords[0] <= 1 && cube_coords[0] >= -1);
    assert(cube_coords[1] <= 1 && cube_coords[1] >= -1);
    assert(cube_coords[2] <= 1 && cube_coords[2] >= -1);
//    std::cout << "Acessing to cube" << std::endl;
    cv::Vec3f face_coords = getFaceCoords(cube_coords);
//    std::cout << "Face coords: " << face_coords << std::endl;
    cv::Vec3f *rowPtr = cube[face_coords[0]].ptr<cv::Vec3f>(face_coords[1]);
    return rowPtr[int(face_coords[2])];
}

cv::Vec3f CubeMap::operator [] (const cv::Vec3f& cube_coords) const
{
    assert(cube_coords[0] <= 1 && cube_coords[0] >= -1);
    assert(cube_coords[1] <= 1 && cube_coords[1] >= -1);
    assert(cube_coords[2] <= 1 && cube_coords[2] >= -1);
//    std::cout << "Acessing to cube" << std::endl;
    cv::Vec3f face_coords = getFaceCoords(cube_coords);
//    std::cout << "Face coords: " << face_coords << std::endl;
    const cv::Vec3f *rowPtr = cube[face_coords[0]].ptr<cv::Vec3f>(face_coords[1]);
    return rowPtr[int(face_coords[2])];
}

cv::Vec3f& CubeMap::operator () (const cv::Vec3f& face_coords)
{
    assert(face_coords[0] <= 5);
    assert(face_coords[1] <= cube_face_size);
    assert(face_coords[1] >= 0);
    assert(face_coords[2] <= cube_face_size);
    assert(face_coords[2] >= 0);
//    std::cout << "Acessing to cube" << std::endl;
    cv::Vec3f *rowPtr = cube[face_coords[0]].ptr<cv::Vec3f>(face_coords[1]);
    return rowPtr[int(face_coords[2])];
}

cv::Vec3f CubeMap::operator () (const cv::Vec3f& face_coords) const
{
    assert(face_coords[0] <= 5);
    assert(face_coords[1] <= cube_face_size && face_coords[1] >= 0);
    assert(face_coords[2] <= cube_face_size && face_coords[2] >= 0);
//    std::cout << "Acessing to cube" << std::endl;
    const cv::Vec3f *rowPtr = cube[face_coords[0]].ptr<cv::Vec3f>(face_coords[1]);
    return rowPtr[int(face_coords[2])];
}

cv::Vec3f CubeMap::getSphericalCoords(const cv::Vec3f& face_coords) const
{
    assert(face_coords[0] <= 5);
    assert(face_coords[1] <= cube_face_size && face_coords[1] >= 0);
    assert(face_coords[2] <= cube_face_size && face_coords[2] >= 0);
    cv::Vec3f aux = getCubeCoords(face_coords);
    cv::normalize(aux, aux);
    return aux;
}

cv::Vec3f CubeMap::getCubeCoords(const cv::Vec3f& face_coords) const
{
    return getCubeCoords(face_coords[0], cv::Vec2f(face_coords[1], face_coords[2]));
}

void CubeMap::resizeCube(unsigned int new_face_size)
{
    cube_face_size = new_face_size;
    for(unsigned int i = 0; i < 6; ++i)
    {
        cv::Size cube_size(new_face_size, new_face_size);
        cv::resize(cube[i], cube[i], cube_size);
    }
}

cv::Vec3f CubeMap::getCubeCoords(unsigned int face, const cv::Vec2f& face_coords) const
{
    assert(face <= 5);
    assert(face_coords[0] <= cube_face_size && face_coords[0] >= 0);
    assert(face_coords[1] <= cube_face_size && face_coords[1] >= 0);

    float corr_to_center = 1/float(cube_face_size);
//    std::cout << "Corr to center: " << corr_to_center << std::endl;
    float x = face_coords[1]/float(cube_face_size)*2 + corr_to_center;
    float y = face_coords[0]/float(cube_face_size)*2 + corr_to_center;

//    if(face == 0)       return cv::Vec3f(1, 1 - y, -1 + x);
//    else if(face == 1)  return cv::Vec3f(-1, 1 - y, 1 - x);
    if(face == 0)       return cv::Vec3f(1, 1 - y, 1 - x);
    else if(face == 1)  return cv::Vec3f(-1, 1 - y, -1 + x);
//    else if(face == 2)  return cv::Vec3f(1 - y, 1, -1 + x);
//    else if(face == 3)  return cv::Vec3f(-1 + y, -1, -1 + x);
    else if(face == 2)  return cv::Vec3f(-1 + x, 1, -1 + y);
    else if(face == 3)  return cv::Vec3f(-1 + x, -1, 1 - y);
//    else if(face == 4)  return cv::Vec3f(1 - x, 1 - y, 1);
//    else                return cv::Vec3f(-1 + x , 1 - y, -1);
    else if(face == 4)  return cv::Vec3f(-1 + x , 1 - y, 1);
    else                return cv::Vec3f(-1 + x, 1 - y, -1);
}

cv::Vec3f CubeMap::getFaceCoords(const cv::Vec3f& cube_coords) const
{
    assert(false);
    assert(cube_coords[0] <= 1 && cube_coords[0] >= -1);
    assert(cube_coords[1] <= 1 && cube_coords[1] >= -1);
    assert(cube_coords[2] <= 1 && cube_coords[2] >= -1);

    float dominant_val = 0;
    int axis = 0;
    for(int i = 0; i < 3; ++i)
    {
        if(dominant_val < std::abs(cube_coords[i]))
        {
            axis = i;
            dominant_val = std::abs(cube_coords[i]);
        }
    }
    int face = 0;
    if(axis == 0 && cube_coords[axis] < 0) face = 1;
    else if(axis == 1 && cube_coords[axis] >= 0) face = 2;
    else if(axis == 1 && cube_coords[axis] < 0) face = 3;
    else if(axis == 2 && cube_coords[axis] >= 0) face = 4;
    else if(axis == 2 && cube_coords[axis] < 0) face = 5;

//    std::cout << "Face: " << face << std::endl;
    cv::Vec3f coords;
//    if(face == 0)       coords = cv::Vec3f(face, 1 - cube_coords[1], 1 + cube_coords[2]);
//    else if(face == 1)  coords = cv::Vec3f(face, 1 - cube_coords[1], 1 - cube_coords[2]);
    if(face == 0)       coords = cv::Vec3f(face, 1 - cube_coords[1], 1 + cube_coords[2]);
    else if(face == 1)  coords = cv::Vec3f(face, 1 - cube_coords[1], 1 - cube_coords[2]);
//    else if(face == 2)  coords = cv::Vec3f(face, 1 - cube_coords[0], 1 + cube_coords[2]);
//    else if(face == 3)  coords = cv::Vec3f(face, 1 + cube_coords[0], 1 + cube_coords[2]);
    else if(face == 2)  coords = cv::Vec3f(face, 1 + cube_coords[2], 1 + cube_coords[0]);
    else if(face == 3)  coords = cv::Vec3f(face, 1 - cube_coords[2], 1 + cube_coords[0]);
//    else if(face == 4)  coords = cv::Vec3f(face, 1 - cube_coords[1], 1 - cube_coords[0]);
//    else                coords = cv::Vec3f(face, 1 - cube_coords[1], 1 + cube_coords[0]);
    else if(face == 4)  coords = cv::Vec3f(face, 1 - cube_coords[1], 1 + cube_coords[0]);
    else                coords = cv::Vec3f(face, 1 - cube_coords[1], 1 + cube_coords[0]);

    coords[1] *= cube_face_size/2;
    coords[2] *= cube_face_size/2;
//    std::cout << "Coords: " << coords << std::endl;
    return coords;
}

//cv::Vec3f get(const cv::Vec3f& cube_coords);
//void put(const cv::Vec3f& cube_coords, const cv::Vec3f& value);

std::vector<cv::Mat> CubeMap::exportIntoImages() const
{
    std::vector<cv::Mat> aux (cube.size());
    for(unsigned int i = 0; i < cube.size(); ++i)
    {
        cube[i].copyTo(aux[i]);
//        cv::imshow("face", (aux[i]));
//        cv::waitKey();
    }
    //cv::flip(aux[5], aux[5], 0);
    return aux;
}

cv::Mat CubeMap::exportCubeCross() const
{
    cv::Mat cube_cross(cube_face_size*4, cube_face_size*3, CV_32FC3, cv::Scalar::all(0));
//    std::cout << "Cross Dims: " << cube_cross.cols << " x " << cube_cross.rows << std::endl;
//    std::cout << "Face Dims: " << cube_face_size << " x " << cube_face_size << std::endl;
    for(unsigned int i = 0; i < 5; ++i)
    {
        cv::Rect rec(cube_face_size*cubes_w[i], cube_face_size*cubes_h[i], cube_face_size, cube_face_size);
//        std::cout << "Rectangle (" << cube_face_size*cubes_w[i] << ", " << cube_face_size*cubes_h[i] << ")" << std::endl;
//        cube_cross(rec) = cube[i];
//        if(i == 2){
//            cv::Mat aux;
//            cv::flip(cube[2], aux, -1);
//            aux.copyTo(cube_cross(rec));
//        }
//        else
//        aux.copyTo(cube_cross(rec));
        cube[i].copyTo(cube_cross(rec));
    }
    std::cout << "last cube" << std::endl;
//    std::cout << "Rectangle (" << cube_face_size*cubes_w[5] << ", " << cube_face_size*cubes_h[5] << ")" << std::endl;
    cv::Rect rec(cube_face_size*cubes_w[5], cube_face_size*cubes_h[5], cube_face_size, cube_face_size);
    cv::Mat aux;
    cube[5].copyTo(aux);

//    cv::imshow("face", (aux));
//    cv::waitKey();

    cv::flip(aux, aux, 0);

//    cv::imshow("face", (aux));
//    cv::waitKey();

    aux.copyTo(cube_cross(rec));
    return cube_cross;
}

unsigned int CubeMap::getFaceSize() const
{
    return cube_face_size;
}
