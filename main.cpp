#include "pastwork/Fecker2008_Demo/utils.hpp"

using namespace std;
using namespace cv;


int main()
{
    // dataset address.
    string data_addr = "../../picture/color_blending_dataset1/";

    // load data.
    ImgPack ref, tar;
    ref.img = imread(data_addr + "warped_reference.png", IMREAD_COLOR);
    ref.mask = imread(data_addr + "result_from_reference.png", IMREAD_GRAYSCALE);
    tar.img = imread(data_addr + "warped_target.png", IMREAD_COLOR);
    tar.mask = imread(data_addr + "result_from_target.png", IMREAD_GRAYSCALE);
    Mat overlap = imread(data_addr + "overlap.png", IMREAD_COLOR);
    Mat Final_Result(ref.img.size(), ref.img.type());
    Mat copy_tar;
    tar.img.copyTo(copy_tar);

    // color space transform, RGB -> YCbCr.
    cvtColor(ref.img, ref.img, CV_BGR2YCrCb);
    cvtColor(tar.img, tar.img, CV_BGR2YCrCb);

    vector<int> map_Array(256, 0);

    for (int channel = 0; channel < 3; channel++)
    {
        // calculate the Density Function of the overlap region.
        CalOverlapDF(overlap, ref, channel);
        CalOverlapDF(overlap, tar, channel);

        map_Array.assign(256, 0);

        // find mapping function and map
        MappingFunction(map_Array, ref, tar, channel);
    }   

    // color space transform, YCbCr -> RGB.
    cvtColor(ref.img, ref.img, CV_YCrCb2BGR);
    cvtColor(tar.img, tar.img, CV_YCrCb2BGR);

    // mask
    ref.img.copyTo(ref.mask_result, ref.mask);
    tar.img.copyTo(tar.mask_result, tar.mask);

    // image fusion.
    addWeighted(ref.mask_result, 1, tar.mask_result, 1, 0, Final_Result);

    // save result.
    imwrite(data_addr + "result/result_of_reference.png", ref.mask_result);
    imwrite(data_addr + "result/result_of_target.png", tar.mask_result);
    imwrite(data_addr + "result/result_of_Fecker.png", Final_Result);

    return 0;
}
