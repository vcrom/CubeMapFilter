#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <QString>
#include <QFileDialog>
#include <QApplication>

#include "cubemap.h"
#include "cubemapfilter.h"

//config params
#define RESIZE_BEFORE
#define CV_WINDOW_FLAGS CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED
std::string image_path = "";
bool show_filt_cross = false;
bool show_filt_faces = false;
bool show_orig_cross = true;
bool export_cross = false;
bool export_faces = false;
int out_face_size = 0;
float filter_angle = 180.0f;
//float pot= 1.0f;
bool range = false;
float pow_min = 1.0f;
float pow_max = 1.0f;
float pow_step = 1.0f;
bool mipmaps = false;
bool fixup = true;
int fixup_band = 1;

//-img uffizi_cross.png -export_cross -out_face_size 64 -pow_range 1 11 -pow_step 2
void printHelp()
{
    std::cout << "Options:" << std::endl;
    std::cout << "-show_filt_cross" << std::endl;
    std::cout << "-show_filt_faces" << std::endl;
    std::cout << "-export_cross" << std::endl;
    std::cout << "-export_faces" << std::endl;
    std::cout << "-out_face_size <size>" << std::endl;
    std::cout << "-pow_range <min> <max> <step>" << std::endl; //min = 1, max = 8192
    std::cout << "-angle <value>" << std::endl;
    std::cout << "-mipmaps <min> <max>" << std::endl;
    std::cout << "-fixup_band <width>" << std::endl; //band arround the cube edges to blurr between them.
}

void printUsage()
{
    std::cout << "Usage: CubeMapFilter -img <image_path> <options>" << std::endl;
    printHelp();
    exit(-1);
}

void handleCmdParams(unsigned int argc, char *argv[])
{
    if(argc < 3) printUsage();
    for(unsigned int i = 1; i < argc; ++i)
    {
        std::string param(argv[i]);
        if(!param.compare("-img"))
        {
            if(i+1 >= argc) printUsage();
            image_path.append(argv[++i]);
        }
        else if(!param.compare("-show_filt_cross"))
        {
            show_filt_cross = true;
        }
        else if(!param.compare("-show_filt_faces"))
        {
            show_filt_faces = true;
        }
        else if(!param.compare("-export_cross"))
        {
            export_cross = true;
        }
        else if(!param.compare("-export_faces"))
        {
            export_faces = true;
        }
        else if(!param.compare("-out_face_size"))
        {
            if(i+1 >= argc) printUsage();
            out_face_size = std::stoi(argv[++i]);
        }
//        else if(!param.compare("-pow_step"))
//        {
//            if(i+1 >= argc) printUsage();
//            pow_step = std::stof(argv[++i]);
//        }
        else if(!param.compare("-angle"))
        {
            if(i+1 >= argc) printUsage();
            filter_angle = std::stof(argv[++i]);
        }
        else if(!param.compare("-pow_range"))
        {
            if(i+3 >= argc) printUsage();
            pow_min = std::stof(argv[++i]);
            pow_max = std::stof(argv[++i]);
            pow_step = std::stof(argv[++i]);
        }
        else if(!param.compare("-mipmaps"))
        {
            if(i+2 >= argc) printUsage();
            mipmaps = true;
            pow_min = std::stof(argv[++i]);
            pow_max = std::stof(argv[++i]);
        }
        else if(!param.compare("-fixup_band"))
        {
            if(i+1 >= argc) printUsage();
            fixup_band = std::stoi(argv[++i]);
        }
        else printUsage();
    }
    assert(pow_min <= pow_max);
}


std::vector<std::vector<cv::Mat> > cubeMapsArrayToImages(const std::vector<CubeMap> &cubes)
{
    std::vector<std::vector<cv::Mat> > images (cubes.size());
    for(unsigned int i = 0; i < cubes.size(); ++i)
    {
        images[i] = cubes[i].exportIntoImages();
//        cv::flip(images[i][5], images[i][5], 0);
    }
    return images;
}

std::vector<cv::Mat> cubeMapsArrayToCrosses(const std::vector<CubeMap> &cubes)
{
    std::vector<cv::Mat> images (cubes.size());
    for(unsigned int i = 0; i < cubes.size(); ++i)
        images[i] = cubes[i].exportCubeCross();
    return images;
}


int main (int argc, char *argv[])
{
    QApplication app(argc, argv);
    handleCmdParams(argc, argv);
    std::cout << "Loading " << image_path << "... ";
    cv::Mat original_cross = cv::imread(image_path);
    if(!original_cross.data)
    {
        std::cout << "Error" << std::endl;
        return -1;
    }
    //convert to 32 Float for homogeinize formats
    original_cross.convertTo(original_cross, CV_32F, 1.0f/255.0f);

    //resize the image if it's needed
    if(out_face_size != 0)
    {
        //image.resize(image, cv::Size2f(out_face_size*3, out_face_size*4));
        cv::resize(original_cross, original_cross, cv::Size(out_face_size*3, out_face_size*4));

    }
    std::cout << "...Loaded" << std::endl;

    if(show_orig_cross)
    {
        cv::namedWindow("Original Cross", CV_WINDOW_FLAGS);
        cv::imshow("Original Cross", original_cross);
        cv::waitKey();
        cv::destroyWindow("Original Cross");
    }

    ////Filtering
    std::vector<CubeMap> filtered_cubes;
    std::vector<float> pows;
    std::vector<std::vector<cv::Mat> > filtered_face_images;
    std::vector<cv::Mat> filtered_cross_images;

    //Load cubemap
    CubeMap cube;
    cube.loadCubeCross(original_cross);
    if(!mipmaps)
    {
//        //Load cubemap
//        CubeMap cube;
//        cube.loadCubeCross(original_cross);

        int len = (pow_max - pow_min)/pow_step;
        pows = std::vector<float> (len+1);
        for(unsigned int i = 0; i < unsigned(len + 1); ++i)
        {
            pows[i] = pow_min + pow_step*i;
            std::cout << "pows[" << i << "] --> " << pows[i] << std::endl;
        }
        filtered_cubes = CubeMapFilter::cosinePowFilterArrayCube(cube, filter_angle, pows);
        cv::waitKey();
//filtered_cross_images = cubeMapsArrayToCrosses(filtered_cubes);
//cv::namedWindow("Bu", CV_WINDOW_FLAGS);
//cv::imshow("Bu", filtered_cross_images[0]);
//cv::waitKey();

        //filtered_cubes[0].resizeCube(int(4));
        filtered_cubes[0] = CubeMapFilter::edgePullFixup(filtered_cubes[0], fixup_band);
        //filtered_cubes.push_back(CubeMapFilter::cosinePowFilterCube(cube, filter_angle, pows[0]));
    }
    else
    {
        //load cube and get cube size and steps
//        CubeMap cube;
//        cube.loadCubeCross(original_cross);
        int face_size = cube.getFaceSize();
        unsigned int steps = std::log2(face_size)+1;
        std::cout << "Steps: " << steps << std::endl;

        //initialize pow
        float step = (pow_max - pow_min)/float(steps-1);
        std::cout << "Step: " << step << std::endl;
        pows = std::vector<float> (steps);
        for(unsigned int i = 0; i < steps; ++i)
        {
            pows[i] = pow_min + step*float(steps - i - 1);
            std::cout << "pows[" << i << "] --> " << pows[i] << std::endl;
        }
//exit(0);
        filtered_cubes = std::vector<CubeMap> (steps);
        cv::Size cross_size = original_cross.size();
        std::cout << "Original size: (" << cross_size.height << ", " << cross_size.width << ")" << std::endl;
        cv::Mat aux_cross;
        std::cout << "aux_cross" << std::endl;
        for(unsigned int i = 0; i < steps; ++i)
        {
//            CubeMap cube_aux;
//            std::cout << "compute step" << std::endl;
            unsigned int step = std::pow(2,i);
//            std::cout << "Compute new size" << std::endl;
            cv::Size new_size(int(cross_size.width/step), int(cross_size.height/step));
            std::cout << "New size: (" << new_size.height << ", " << new_size.width << ")" << std::endl;
//            std::cout << "Resize" << std::endl;
            #ifdef RESIZE_BEFORE
                cv::resize(original_cross, aux_cross, new_size);
            #endif
//            std::cout << "Load aux cross" << std::endl;
            CubeMap aux_cube;
            #ifdef RESIZE_BEFORE
                aux_cube.loadCubeCross(aux_cross);
            #else
                aux_cube.loadCubeCross(original_cross);
            #endif
//            aux_cube.loadCubeCross(original_cross);
//            std::cout << "Filter!!" << std::endl;
            filtered_cubes[i] = CubeMapFilter::cosinePowFilterCube(aux_cube, filter_angle, pows[i]);
            std::cout << "---Face Size: " << int((cross_size.width/step)/3) << std::endl;
            #ifndef RESIZE_BEFORE
                filtered_cubes[i].resizeCube(int((cross_size.width/step)/3));
            #endif
            std::cout << "---Cube Resized." << std::endl;
            std::cout << "Fixup Band!!!!!" << std::endl;
            filtered_cubes[i] = CubeMapFilter::edgePullFixup(filtered_cubes[i], fixup_band);
        }
    }
std::cout << "Cube Filtered!" << std::endl;
    if(show_filt_faces || export_faces) filtered_face_images = cubeMapsArrayToImages(filtered_cubes);
std::cout << "Cube Images!" << std::endl;
    if(show_filt_cross || export_cross) filtered_cross_images = cubeMapsArrayToCrosses(filtered_cubes);
std::cout << "Cube Crosses!" << std::endl;

////Filtering
    if(show_filt_faces)
    {
        for(unsigned int i = 0; i < filtered_face_images.size(); ++i)
        {
            std::string window_name = "Faces from level "+std::to_string(i);
            cv::namedWindow(window_name, CV_WINDOW_FLAGS);
            for(unsigned int j = 0; j < filtered_face_images[0].size(); ++j)
            {
                cv::imshow(window_name, filtered_face_images[i][j]);
                cv::waitKey();
            }
            cv::destroyWindow(window_name);
        }
    }

    if(show_filt_cross)
    {
        for(unsigned int i = 0; i < filtered_cross_images.size(); ++i)
        {
            std::string window_name = "Cross from level "+std::to_string(i);
            cv::namedWindow(window_name, CV_WINDOW_FLAGS);
            cv::imshow(window_name, filtered_cross_images[i]);
            cv::waitKey();
            cv::destroyWindow(window_name);
        }
    }

    if(export_cross)
    {
        cv::Mat export_cross;
        QString dir = QFileDialog::getExistingDirectory(0, QObject::tr("Save Cube Cross"), "./", QFileDialog::ShowDirsOnly);
        if(!dir.isEmpty())
        {
            std::cout << "Saving crosses into: " << dir.toStdString() << std::endl;
            //QString path = QFileDialog::getSaveFileName(0, QObject::tr("Save cube cross"), "./", QObject::tr("PNG Files (*.png);;All files (*)"));
            for(unsigned int i = 0; i < filtered_cross_images.size(); ++i)
            {
                ///FIXME
                //convert format to be RGB this must be modified if the image was hdr
                filtered_cross_images[i].convertTo(export_cross, CV_8U, 255.0f);
                try {
                    cv::imwrite(dir.toStdString()+"/filtered_cross_level_"+std::to_string(i)+".png", export_cross);
                }
                catch (std::runtime_error& ex) {
                    std::cerr << "Exception exporting cross image: %s\n" <<  ex.what();
                    return 1;
                }
            }
        }
        else std::cout << "Not selected Directory" << std::endl;
    }

    if(export_faces)
    {
        cv::Mat export_face;
        QString dir = QFileDialog::getExistingDirectory(0, QObject::tr("Save Cube Faces"), "./", QFileDialog::ShowDirsOnly);
        std::cout << "Saving faces into: " << dir.toStdString() << std::endl;
        for(unsigned int i = 0; i < filtered_face_images.size(); ++i)
        {
            std::string aux = dir.toStdString()+"/faces_level"+std::to_string(i);
//            std::cout << "Saving faces into: " << aux << std::endl;
            QDir faces_dir(QString::fromStdString(aux));
            if (!faces_dir.exists()) faces_dir.mkpath(".");

            for(unsigned int j = 0; j < filtered_face_images[0].size(); ++j)
            {
//                std::cout << "Exporting: " << faces_dir.absolutePath().toStdString()+"/i_c0"+std::to_string(j)+".png" << std::endl;
                ///FIXME
                //convert format to be RGB this must be modified if the image was hdr
                filtered_face_images[i][j].convertTo(export_face, CV_8U, 255.0f);
//                if(j == 5) cv::flip(export_face, export_face, 0);
                cv::imwrite(faces_dir.absolutePath().toStdString()+"/i_c0"+std::to_string(j)+".png", export_face);

            }
        }
    }

////Show import and export
//    std::vector<cv::Mat> images = cube.exportIntoImages();
//    for(unsigned int i = 0; i < images.size(); ++i)
//    {
//        cv::imshow("Image win", images[i]);
//        cv::waitKey();
//    }

//    cv::Mat cross = cube.exportCubeCross();
//    cv::imshow("Image win", cross);
//    cv::waitKey();

//////Test accessors
//    for(int face = 0; face < 6; ++face)
//    {
//        std::cout << "FACE " << face << std::endl;
//        for(int i = 0; i < 4; ++i)
//        {
//            for(int j = 0; j < 4; ++j)
//            {
//                cv::Vec3f face_coords = cv::Vec3f(face, i, j);
//                cv::Vec3f aux_v = cube.getCubeCoords(face_coords);
//                std::cout << "Face Coords: " << face_coords << ", Cube coords: " << aux_v << std::endl;
//            }
//        }
//    }

//////Cross load and export
//        std::vector<cv::Mat> images = cube.exportIntoImages();
//        for(unsigned int i = 0; i < images.size(); ++i)
//        {
//            cv::imshow("Image win", images[i]);
//            cv::waitKey();
//        }

//        cv::Mat cross = cube.exportCubeCross();
//        cv::imshow("Image win", cross);
//        cv::waitKey();

////Black square
    //    int face_size = cube.getFaceSize();
    //    for(int i = 0; i < 6; ++i)
    //    {
    //        for(int j = 0; j < 50; ++j )
    //        {
    //            for(int k = 0; k < 20; ++k)
    //            {
    //                cube(cv::Vec3f(i, j, k)) = cv::Vec3f(0,0,0);
    //            }
    //        }
    //    }
}
