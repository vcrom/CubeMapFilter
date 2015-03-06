#include "cubemapfilter.h"

const unsigned int CubeMapFilter::oposite_face[6] = {1, 0, 3, 2, 5, 4};

CubeMapFilter::CubeMapFilter()
{
}

std::vector<std::vector<std::vector<cv::Vec3f> > > CubeMapFilter::fillSphericalCoordsTable(const CubeMap& src)
{
    unsigned int cube_face_size = src.getFaceSize();
    std::vector<std::vector<std::vector<cv::Vec3f> > > spherical_coords_table (6, std::vector<std::vector<cv::Vec3f> > (cube_face_size, std::vector<cv::Vec3f> (cube_face_size, cv::Vec3f(0,0,0))));
    #pragma omp parallel for schedule(dynamic) collapse(2)
    for(unsigned int face = 0; face < 6; ++face)
        for(unsigned int i = 0; i < cube_face_size; ++i)
            for(unsigned int j = 0; j < cube_face_size; ++j)
                spherical_coords_table[face][i][j] = src.getSphericalCoords(cv::Vec3f(face, i, j));
    return spherical_coords_table;

}

CubeMap CubeMapFilter::cosineFilterCube(CubeMap& src, float angle)
{
    return cosinePowFilterCube(src, angle, 1.0f);
}

//CubeMap CubeMapFilter::cosinePowFilterCube(CubeMap& src, float angle, float pow)
CubeMap CubeMapFilter::cosinePowFilterCube(const CubeMap& src, float angle, float pow)
{
    float cos_tresh = std::cos(degToRad(angle));
    unsigned int cube_face_size = src.getFaceSize();
    CubeMap filtered(cube_face_size);
    std::vector<std::vector<std::vector<cv::Vec3f> > > spherical_coords_table = fillSphericalCoordsTable(src);

    #pragma omp parallel for schedule(dynamic) collapse(2)
    for(unsigned int face = 0; face < 6; ++face)
    {
        for(unsigned int i = 0; i < cube_face_size; ++i)
        {
            int tid = omp_get_thread_num();
            #pragma omp critical
            {
            std::cout << "Thread: " << tid << ", Face: " << face << ", Row: " << i << std::endl;
            }
            for(unsigned int j = 0; j < cube_face_size; ++j)
            {
                cv::Vec3f color_acum = cv::Vec3f(0, 0, 0);
                float total_weight = 0.0f;
                cv::Vec3f ref_norm_vect = spherical_coords_table[face][i][j];
                for(unsigned int inner_face = 0; inner_face < 6; ++inner_face)
                {
                    //skip oposite face
                    if(oposite_face[face] == inner_face) continue;
                    for(unsigned int k = 0; k < cube_face_size; ++k)
                    {
                        for(unsigned int l = 0; l < cube_face_size; ++l)
                        {
                            cv::Vec3f loc_norm_vect = spherical_coords_table[inner_face][k][l];
                            float cosine = ref_norm_vect.dot(loc_norm_vect);
                            if(cosine < 0) continue;// = 0;//from cubeMapGen
                            if(cosine >= cos_tresh)
                            {
                                float weight = std::pow(cosine, pow);
                                color_acum += src(cv::Vec3f(inner_face, k, l)) * weight;
                                total_weight += weight;
                            }
                        }
                    }
                }
                color_acum /= total_weight;
                filtered(cv::Vec3f(face, i, j)) = color_acum;
//                cv::Mat cross = filtered.exportCubeCross();
//                cv::imshow("Image win", cross);
//                cv::waitKey();
            }
        }
    }

    return filtered;
}


std::vector<CubeMap> CubeMapFilter::cosinePowFilterArrayCube(CubeMap& src, float angle, const std::vector<float> &pows)
{
    //cv::namedWindow("Cross step", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);

    float cos_tresh = std::cos(degToRad(angle));
    unsigned int cube_face_size = src.getFaceSize();
    std::vector<CubeMap> filtered_cubes (pows.size());//, CubeMap(cube_face_size));
    for(unsigned int i = 0; i < pows.size(); ++i)
        filtered_cubes[i] = CubeMap(cube_face_size);

    std::vector<std::vector<std::vector<cv::Vec3f> > > spherical_coords_table = fillSphericalCoordsTable(src);
    #pragma omp parallel for schedule(dynamic) collapse(2) //private(color_acums) private(sum_weight)
    for(unsigned int face = 0; face < 6; ++face)
    {
        for(unsigned int i = 0; i < cube_face_size; ++i)
        {
            int tid = omp_get_thread_num();
            #pragma omp critical
            {
                std::cout << "Thread: " << tid << ", Face: " << face << ", Row: " << i << std::endl;
            }
            for(unsigned int j = 0; j < cube_face_size; ++j)
            {
                std::vector<cv::Vec3f> color_acums (pows.size(), cv::Vec3f(0, 0, 0));
                std::vector<float> sum_weight (pows.size(), 0.0f);
                cv::Vec3f ref_norm_vect = spherical_coords_table[face][i][j];
                for(unsigned int inner_face = 0; inner_face < 6; ++inner_face)
                {
                    //skip oposite face
                    if(oposite_face[face] == inner_face) continue;
                    for(unsigned int k = 0; k < cube_face_size; ++k)
                    {
                        for(unsigned int l = 0; l < cube_face_size; ++l)
                        {
                            cv::Vec3f loc_norm_vect = spherical_coords_table[inner_face][k][l];
                            float cosine = ref_norm_vect.dot(loc_norm_vect);
                            if(cosine < 0) continue;//from cubeMapGen
                            if(cosine >= cos_tresh)
                            {
                                for(unsigned int w = 0; w < pows.size(); ++w)
                                {
                                    float weight = std::pow(cosine, pows[w]);
                                    color_acums[w] += src(cv::Vec3f(inner_face, k, l)) * weight;
                                    sum_weight[w] += weight;
                                }
                            }
                        }
                    }
                }
                for(unsigned int w = 0; w < pows.size(); ++w)
                {
                    color_acums[w] /= sum_weight[w];
                    filtered_cubes[w](cv::Vec3f(face, i, j)) = color_acums[w];
                }
//                cv::Mat cross = filtered_cubes[0].exportCubeCross();
//                cv::imshow("Cross step", cross);
//                cv::waitKey();
            }
        }
    }

//    for(unsigned int i = 0; i < filtered_cubes.size(); ++i)
//        filtered_cubes[i] = edgePullFixup(filtered_cubes[i], 4);
    return filtered_cubes;
}

float CubeMapFilter::pixelIntensity(const cv::Vec3f &vec)
{
    return (vec[0] + vec[1] + vec[2])/3;
}

CubeMap CubeMapFilter::edgeSmoothFixup(CubeMap& src, unsigned int len_fixup_band)
{
    return edgeFixup(src, len_fixup_band, false);
}

CubeMap CubeMapFilter::edgePullFixup(CubeMap& src, unsigned int len_fixup_band)
{
    return edgeFixup(src, len_fixup_band, true);
}

CubeMap CubeMapFilter::edgeFixup(CubeMap src, unsigned int len_fixup_band, bool pull)
{
    unsigned int cube_face_size = src.getFaceSize();
    unsigned int fixup_band = std::min(cube_face_size, len_fixup_band);

    //compute linear falloff
    std::vector<float> falloff_vec (len_fixup_band, 1.0f);
    for(unsigned int i = 0; i < fixup_band; ++i)
    {
        falloff_vec[i] -= float(i)/float(fixup_band);
        std::cout << "Fall[" << i << "] = " << falloff_vec[i] << std::endl;
    }
    std::cout << "Face size: " << cube_face_size << std::endl;
    std::cout << "Fix up: " << fixup_band << std::endl;
    ///Vertices
    std::vector<std::vector<cv::Vec3i> >vertices =
    {
        //vert 0
        std::vector<cv::Vec3i>
        {
            cv::Vec3i(4, cube_face_size-1, cube_face_size-1), cv::Vec3i(0, cube_face_size-1, 0), cv::Vec3i(3, 0, cube_face_size-1)
        },
        //vert 1
        std::vector<cv::Vec3i>
        {
            cv::Vec3i(4, 0, cube_face_size-1), cv::Vec3i(0, 0, 0), cv::Vec3i(2, cube_face_size-1, cube_face_size-1)
        },
        //vert 2
        std::vector<cv::Vec3i>
        {
            cv::Vec3i(4, 0, 0), cv::Vec3i(2, cube_face_size-1, 0), cv::Vec3i(1, 0, cube_face_size-1)
        },
        //vert 3
        std::vector<cv::Vec3i>
        {
            cv::Vec3i(4, cube_face_size-1, 0), cv::Vec3i(3, 0, 0), cv::Vec3i(1, cube_face_size-1, cube_face_size-1)
        },
        //vert 4
        std::vector<cv::Vec3i>
        {
            cv::Vec3i(5, cube_face_size-1, 0), cv::Vec3i(3, cube_face_size-1, 0), cv::Vec3i(1, cube_face_size-1, 0)
        },
        //vert 5
        std::vector<cv::Vec3i>
        {
            cv::Vec3i(5, cube_face_size-1, cube_face_size-1), cv::Vec3i(3, cube_face_size-1, cube_face_size-1), cv::Vec3i(0, cube_face_size-1, cube_face_size-1)
        },
        //vert 6
        std::vector<cv::Vec3i>
        {
            cv::Vec3i(5, 0, 0), cv::Vec3i(1, 0, 0), cv::Vec3i(2, 0, 0)
        },
        //vert 7
        std::vector<cv::Vec3i>
        {
            cv::Vec3i(5, 0, cube_face_size-1), cv::Vec3i(0, 0, cube_face_size-1), cv::Vec3i(2, 0, cube_face_size-1)
        }
    };

    std::vector<cv::Vec3f> pix_intensity(3);
    for(unsigned int i = 0; i < vertices.size(); ++i)
    {
        pix_intensity[0] = src(vertices[i][0]);
        pix_intensity[1] = src(vertices[i][1]);
        pix_intensity[2] = src(vertices[i][2]);
        cv::Vec3f mean_intensity = (pix_intensity[0] + pix_intensity[1] + pix_intensity[2])/3;
//        std::cout << "Pix mean: " << mean_intensity << std::endl;
        for(unsigned int j = 0; j < 3; ++j)
        {
            cv::Vec3f delta_intensity;
            float dist = 0;
            if(pull)
            {
                delta_intensity = mean_intensity - pix_intensity[j];
                dist = cv::norm(delta_intensity);
            }
            cv::Vec2i signs = cv::Vec2i(1, 1);
            cv::Vec3f vert_coords = vertices[i][j];
            if(vert_coords[1] > 0) signs[0] = -1;
            if(vert_coords[2] > 0) signs[1] = -1;
            unsigned int k = 0;
//            for(unsigned int k = 0; k < fixup_band; ++k)
//            {
                cv::Vec2i aux = signs*int(k);
                cv::Vec3f coords1 = cv::Vec3f(vertices[i][j][0], vertices[i][j][1]+aux[0], vertices[i][j][2]);
                if(!pull) delta_intensity = mean_intensity - src(coords1);
                else
                {
                    cv::normalize(mean_intensity - src(coords1), delta_intensity);
                    delta_intensity *= dist;
                }
                src(coords1) += delta_intensity*falloff_vec[k];
                if(k > 0)
                {
                    cv::Vec3f coords2 = cv::Vec3f(vertices[i][j][0], vertices[i][j][1], vertices[i][j][2]+aux[1]);
                    if(!pull) delta_intensity = mean_intensity - src(coords2);
                    else
                    {
                        cv::normalize(mean_intensity - src(coords2), delta_intensity);
                        delta_intensity *= dist;
                    }
                    src(coords2) += delta_intensity*falloff_vec[k];
                }
//            }
        }
    }

///////////Edges
    std::vector<std::vector<cv::Vec3i> >edges =
    {
        //edge0
        std::vector<cv::Vec3i>
        {
            cv::Vec3i(4, 0, cube_face_size-1), cv::Vec3i(0, 0, 0)
        },
        //edge1
        std::vector<cv::Vec3i>
        {
            cv::Vec3i(4, 0, 0), cv::Vec3i(2, cube_face_size-1, 0)
        },
        //edge2
        std::vector<cv::Vec3i>
        {
            cv::Vec3i(4, 0, 0), cv::Vec3i(1, 0, cube_face_size-1)
        },
        //edge3
        std::vector<cv::Vec3i>
        {
            cv::Vec3i(4, cube_face_size-1, 0), cv::Vec3i(3, 0, 0)
        },
        //edge4
        std::vector<cv::Vec3i>
        {
            cv::Vec3i(3, 0, 0), cv::Vec3i(1, cube_face_size-1, cube_face_size-1)
        },
        //edge5
        std::vector<cv::Vec3i>
        {
            cv::Vec3i(3, 0, cube_face_size-1), cv::Vec3i(0, cube_face_size-1, 0)
        },
        //edge6
        std::vector<cv::Vec3i>
        {
            cv::Vec3i(2, cube_face_size-1, cube_face_size-1), cv::Vec3i(0, 0, 0)
        },
        //edge7
        std::vector<cv::Vec3i>
        {
            cv::Vec3i(2, cube_face_size-1, 0), cv::Vec3i(1, 0, cube_face_size-1)
        },
        //edge8
        std::vector<cv::Vec3i>
        {
            cv::Vec3i(5, cube_face_size-1, cube_face_size-1), cv::Vec3i(0, cube_face_size-1, cube_face_size-1)
        },
        //edge9
        std::vector<cv::Vec3i>
        {
            cv::Vec3i(5, cube_face_size-1, cube_face_size-1), cv::Vec3i(3, cube_face_size-1, cube_face_size-1)
        },
        //edge10
        std::vector<cv::Vec3i>
        {
            cv::Vec3i(5, 0, 0), cv::Vec3i(1, 0, 0)
        },
        //edge11
        std::vector<cv::Vec3i>
        {
            cv::Vec3i(5, 0, 0), cv::Vec3i(2, 0, 0)
        }
    };

    std::vector<std::vector<cv::Vec2i> >edge_dirs =
    {
        //edge0
        std::vector<cv::Vec2i>
        {
            cv::Vec2i(1, 0)
        },
        //edge1
        std::vector<cv::Vec2i>
        {
            cv::Vec2i(0, 1)
        },
        //edge2
        std::vector<cv::Vec2i>
        {
            cv::Vec2i(1, 0)
        },
        //edge3
        std::vector<cv::Vec2i>
        {
            cv::Vec2i(0, 1)
        },
        //edge4
        std::vector<cv::Vec2i>
        {
            cv::Vec2i(1, 0),
            cv::Vec2i(0, -1)
        },
        //edge5
        std::vector<cv::Vec2i>
        {
            cv::Vec2i(1, 0),
            cv::Vec2i(0, 1)
        },
        //edge6
        std::vector<cv::Vec2i>
        {
            cv::Vec2i(-1, 0),
            cv::Vec2i(0, 1)
        },
        //edge7
        std::vector<cv::Vec2i>
        {
            cv::Vec2i(-1, 0),
            cv::Vec2i(0, -1)
        },
        //edge8
        std::vector<cv::Vec2i>
        {
            cv::Vec2i(-1, 0)
        },
        //edge9
        std::vector<cv::Vec2i>
        {
            cv::Vec2i(0, -1)
        },
        //edge10
        std::vector<cv::Vec2i>
        {
            cv::Vec2i(1, 0)
        },
        //edge11
        std::vector<cv::Vec2i>
        {
            cv::Vec2i(0, 1)
        }
    };

    pix_intensity = std::vector<cv::Vec3f>(2);
    for(unsigned int i = 0; i < edges.size(); ++i)
    {
        cv::Vec3i point1 = edges[i][0];
        cv::Vec3i point2 = edges[i][1];
        //Obtain edge dirs
        cv::Vec2i dir1 = edge_dirs[i][0];
        cv::Vec2i dir2 = dir1;
        if(edge_dirs[i].size() > 1) dir2 = edge_dirs[i][1];
        //Obtain propagation dirs
        cv::Vec2i pdir1 = cv::Vec2i(dir1[1], -dir1[0]);
        if(point1[1]+pdir1[0] < 0 || point1[1]+pdir1[0] >= int(cube_face_size) || point1[2]+pdir1[1] < 0 || point1[2]+pdir1[1] >= int(cube_face_size)) pdir1 *= -1;
        cv::Vec2i pdir2 = cv::Vec2i(dir2[1], -dir2[0]);
        if(point2[1]+pdir2[0] < 0 || point2[1]+pdir2[0] >= int(cube_face_size) || point2[2]+pdir2[1] < 0 || point2[2]+pdir2[1] >= int(cube_face_size)) pdir2 *= -1;

//std::cout << "dir1: "  << dir1 << std::endl;
//std::cout << "pdir1: "  << pdir1 << std::endl;
//std::cout << "dir2: "  << dir2 << std::endl;
//std::cout << "pdir2: "  << pdir2 << std::endl;

        for(unsigned int j = 1; j < cube_face_size-1; ++j)
        {
            cv::Vec2i aux1 = dir1*int(j);
            cv::Vec3f coords1 = cv::Vec3f(point1[0], point1[1]+aux1[0], point1[2]+aux1[1]);
            cv::Vec2i aux2 = dir2*int(j);
            cv::Vec3f coords2 = cv::Vec3f(point2[0], point2[1]+aux2[0], point2[2]+aux2[1]);
            pix_intensity[0] = src(coords1);
            pix_intensity[1] = src(coords2);
            cv::Vec3f mean_intensity = (pix_intensity[0] + pix_intensity[1])/2.0;
            cv::Vec3f delta_intensity1, delta_intensity2;
            float dist1, dist2;
            if(pull)
            {
                delta_intensity1 = mean_intensity - pix_intensity[0];
                dist1 = cv::norm(delta_intensity1);
                delta_intensity2 = mean_intensity - pix_intensity[1];
                dist2 = cv::norm(delta_intensity2);
            }

//            src(coords1) = cv::Vec3f(1,1,1);
//            src(coords2) = cv::Vec3f(0,0,0);

            for(unsigned int k = 0; k < fixup_band; ++k)
            {
                cv::Vec2i aux1 = pdir1*int(k);
                cv::Vec3f coords1_despl = cv::Vec3f(coords1[0], coords1[1]+aux1[0], coords1[2]+aux1[1]);
                if(!pull) delta_intensity1 = mean_intensity - src(coords1_despl);
                else
                {
                    cv::normalize(mean_intensity - src(coords1_despl), delta_intensity1);
                    delta_intensity1 *= dist1;
                }
                src(coords1_despl) += delta_intensity1*falloff_vec[k];

                cv::Vec2i aux2 = pdir2*int(k);
                cv::Vec3f coords2_despl = cv::Vec3f(coords2[0], coords2[1]+aux2[0], coords2[2]+aux2[1]);
                if(!pull) delta_intensity2 = mean_intensity - src(coords2_despl);
                else
                {
                    cv::normalize(mean_intensity - src(coords2_despl), delta_intensity2);
                    delta_intensity2 *= dist2;
                }
                src(coords2_despl) += delta_intensity2*falloff_vec[k];
            }

        }
    }

    return src;
}
