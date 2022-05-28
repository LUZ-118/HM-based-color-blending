#include "Utils.hpp"

using namespace std;
using namespace cv;

int main()
{
    // dataset address.
    string data_addr = "image/";

    // load data.
    ImgPack ref, tar;
    ref.img = imread(data_addr + "warped_reference.png");
    ref.mask = imread(data_addr + "result_from_reference.png");
    tar.img = imread(data_addr + "warped_target.png");
    tar.mask = imread(data_addr + "result_from_target.png");
    Mat overlap = imread(data_addr + "overlap.png");
    Mat Final_Result(ref.img.size(), ref.img.type());
    Mat copy_tar;
    tar.img.copyTo(copy_tar);

    // color space transform, RGB -> YUV.
    cvtColor(ref.img, ref.img, CV_RGB2YUV);
    cvtColor(tar.img, tar.img, CV_RGB2YUV);

    vector<int> map_Array(256, 0);

    for (int channel = 0; channel < 3; channel++)
    {
        // calculate the Density Function of the overlap region.
        CalOverlapDF(overlap, ref, channel);
        CalOverlapDF(overlap, tar, channel);

        map_Array.assign(256, 0);

        // mapping.
        MappingFunction(map_Array, ref, tar, channel);
    }   

    // color space transform, YUV -> RGB.
    cvtColor(ref.img, ref.img, CV_YUV2RGB);
    cvtColor(tar.img, tar.img, CV_YUV2RGB);

    // image mask.
    ref.img.copyTo(ref.mask_result, ref.mask);
    tar.img.copyTo(tar.mask_result, tar.mask);

    // image fusion.
    addWeighted(ref.mask_result, 1, tar.mask_result, 1, 0, Final_Result);

    // save result.
    imwrite(data_addr + "result/result_of_reference.png", ref.mask_result);
    imwrite(data_addr + "result/result_of_target.png", tar.mask_result);
    imwrite(data_addr + "result/result_of_Fecker.png", Final_Result);

    /*// show 2 image in the same window.
    Mat big;
    big.create(ref.img.rows, ref.img.cols*2+20, ref.img.type());
    Rect roi1(0, 0, ref.img.cols, ref.img.rows);
    Rect roi2(ref.img.cols+20, 0, ref.img.cols, ref.img.rows);

    Final_Result.copyTo(big(roi1));
    copy_tar.copyTo(big(roi2));

    namedWindow("result", CV_WINDOW_NORMAL);
    moveWindow("result",0 ,0);
    imshow("result", big);
    waitKey();
    */

    return 0;
}
