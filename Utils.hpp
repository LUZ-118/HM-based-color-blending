#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <string>
#include <vector>

using namespace std;
using namespace cv;


struct ImgPack
{
    public:
        Mat img;
        Mat mask;
        Mat mask_result;
        vector<float> PDF;
        vector<float> CDF;
};

void CalOverlapDF(Mat &overlap, ImgPack &src, int channel)
{
    // initialize vector
    src.PDF.assign(256, 0);
    src.CDF.assign(256, 0);

    // calculate number of pixels in the overlap region.
    int overlap_pixels = 0;

    // Probability Density Dunction.
    for(int i=0;i<src.img.rows;i++)
    {
        for(int j=0;j<src.img.cols;j++)
        {
            if((int)overlap.at<uchar>(i,j) == 255)
            {
                src.PDF[(int)src.img.at<Vec3b>(i,j)[channel]]++;
                overlap_pixels++;
            }  
        }
    }
    
    // float sum=0;
    src.CDF = src.PDF;

    // Cumulative Density Dunction.
    for(int i=1;i<256;i++)
        src.CDF[i] += src.CDF[i-1];
    

    for(int i=0;i<256;i++)
    {
        src.PDF[i] /= overlap_pixels;
        src.CDF[i] /= overlap_pixels;
        // sum += src.PDF[i];
    }
};

// compute the mapping function,by which we can update tar_img's color style.
void MappingFunction(vector<int> &map_Array, ImgPack &ref, ImgPack &tar, int channel)
{
    // for record the first x & y that meet the condition,
    // create a switch(flag) to do that.
    bool flag = false;
    int temp_x = -100, temp_y = -100;
    
    for(int i=0;i<256;i++)
    {
        for(int j=0;j<256;j++)
        {
            if(ref.CDF[j] >= tar.CDF[i])
            {
                if(flag == false && (j-1)>=0)
                {
                    // switch on.
                    flag = true;
                    temp_x = i;
                    temp_y = j-1;
                }

                // make sure (j-1) cannot over or under the color depth(8bits,
                // 0-255) value by use function saturate_cast<uchar>.
                map_Array[i]=(int)saturate_cast<uchar>(j-1); 
                break;
            }

            // if we haven't any value can map,
            // assign the previous one.
            map_Array[i] = map_Array[i-1];
        }
    }

    cout<<"tempx,y: "<<temp_x<<' '<<temp_y<<endl;

    // if we haven't any value can map,
    // assign the first one which we mapped(temp_y).
    for(int i=temp_x;i>=0;i--)  
    {
        map_Array[i] = temp_y;
    }

    // find the center of mass in range [0, M[0]].
    float sum1=0, sum2=0;
    for(int i=0; i<=map_Array[0]; i++)
    {
        sum1 += ref.PDF[i];
        sum2 += ref.PDF[i]*i;
    }
    map_Array[0] = sum2 / sum1;
    
    // find the center of mass in range [M[255], 255].
    sum1=0, sum2=0;
    for(int i=255; i>=map_Array[255]; i--)
    {
        sum1 += ref.PDF[i];
        sum2 += ref.PDF[i]*i;
    }
    map_Array[255] = sum2 / sum1; 

    // check mapping function.
    int n = 0;
    cout<<"\nchannel: "<<channel<<endl;
    for(auto &item : map_Array)
    {
        cout<<n<<" -> "<<item<<endl;
        n++;
    }
    
    // mapping.
    for(int i=0;i<tar.img.rows;i++)
    {
        for(int j=0;j<tar.img.cols;j++)
        {
            tar.img.at<Vec3b>(i,j)[channel] = map_Array[(int)tar.img.at<Vec3b>(i,j)[channel]];
        }
    }
};