#pragma once

#include <opencv2/core/version.hpp>

#if CV_MAJOR_VERSION >= 3
#    include <opencv2/imgcodecs.hpp>
#else
#    include <opencv2/highgui/highgui.hpp>
#endif

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vpi/OpenCVInterop.hpp>

#include <vpi/Image.h>
#include <vpi/LensDistortionModels.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/Remap.h>

#include <iostream>
#include <sstream>

#define CHECK_STATUS(STMT)                                    \
    do                                                        \
    {                                                         \
        VPIStatus status = (STMT);                            \
        if (status != VPI_SUCCESS)                            \
        {                                                     \
            char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];       \
            vpiGetLastStatusMessage(buffer, sizeof(buffer));  \
            std::ostringstream ss;                            \
            ss << vpiStatusGetName(status) << ": " << buffer; \
            throw std::runtime_error(ss.str());               \
        }                                                     \
    } while (0);

class VpiRectify{
public:
    VpiRectify(int width, int height, cv::Matx<double, 3, 3> camMatrix, std::vector<double> coeffs){
        width_ = (int16_t)width;
        height_ = (int16_t)height;

        // Allocate a dense map.
        map.grid.numHorizRegions  = 1;
        map.grid.numVertRegions   = 1;
        map.grid.regionWidth[0]   = width_;
        map.grid.regionHeight[0]  = height_;
        map.grid.horizInterval[0] = 1;
        map.grid.vertInterval[0]  = 1;
        CHECK_STATUS(vpiWarpMapAllocData(&map));

        // Initialize the fisheye lens model with the coefficients given by calibration procedure.
        distModel.mapping                       = VPI_FISHEYE_EQUIDISTANT;
        distModel.k1                            = coeffs[0];
        distModel.k2                            = coeffs[1];
        distModel.k3                            = coeffs[2];
        distModel.k4                            = coeffs[3];

        if (std::abs(coeffs[0]) <= 1e-7 &&
            std::abs(coeffs[1]) <= 1e-7 &&
            std::abs(coeffs[2]) <= 1e-7 &&
            std::abs(coeffs[3]) <= 1e-7){
                do_not_rectify_ = true;
                return;
            }

        // Fill up the camera intrinsic parameters given by camera calibration procedure.
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                K[i][j] = camMatrix(i, j);
            }
        }

        // Camera extrinsics is be identity.
        X[0][0] = X[1][1] = X[2][2] = 1;

        // Generate a warp map to undistort an image taken from fisheye lens with
        // given parameters calculated above.
        vpiWarpMapGenerateFromFisheyeLensDistortionModel(K, X, K, &distModel, &map);

        // Create the Remap payload for undistortion given the map generated above.
        CHECK_STATUS(vpiCreateRemap(VPI_BACKEND_CUDA, &map, &remap));

        // Now that the remap payload is created, we can destroy the warp map.
        vpiWarpMapFreeData(&map);

        // Create a stream where operations will take place. We're using CUDA
        // processing.
        CHECK_STATUS(vpiStreamCreate(VPI_BACKEND_CUDA, &stream));

        CHECK_STATUS(vpiImageCreate(width_, height_, VPI_IMAGE_FORMAT_Y8_ER, VPI_BACKEND_CUDA, &tmpIn));
        // CHECK_STATUS(vpiImageCreate(width_, height_, VPI_IMAGE_FORMAT_U8, VPI_BACKEND_CUDA, &tmpOut));

    }
    ~VpiRectify(){
        vpiStreamDestroy(stream);
        vpiPayloadDestroy(remap);
        vpiImageDestroy(vimg);
    }

    void compute(cv::Mat &cvImage){
        assert(!cvImage.empty());
        if (do_not_rectify_){
            return;
        }

        // Wrap it into a VPIImage
        if (vimg == nullptr)
        {
            // Now create a VPIImage that wraps it.
            CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cvImage, VPI_IMAGE_FORMAT_U8, VPI_BACKEND_CUDA, &vimg));
            // vpiImageCreateWrapperOpenCVMat()
        }
        else
        {
            CHECK_STATUS(vpiImageSetWrappedOpenCVMat(vimg, cvImage));
        }

        CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, vimg, tmpIn, NULL));

        // Undistorts the input image.
        CHECK_STATUS(vpiSubmitRemap(stream, VPI_BACKEND_CUDA, remap, tmpIn, vimg, VPI_INTERP_NEAREST,
                                    VPI_BORDER_CLAMP, 0));

        // Wait until conversion finishes.
        CHECK_STATUS(vpiStreamSync(stream));

        // Since vimg is wrapping the OpenCV image, the result is already there.
        // We just have to save it to disk.
    }

    void compute(cv::Mat &cvImage, cv::Mat &cvImageOut){
        assert(!cvImage.empty());
        if (do_not_rectify_){
            return;
        }

        // Wrap it into a VPIImage
        if (vimg == nullptr)
        {
            CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cvImage, VPI_IMAGE_FORMAT_BGR8, VPI_BACKEND_CUDA, &vimg));
        }
        else
        {
            CHECK_STATUS(vpiImageSetWrappedOpenCVMat(vimg, cvImage));
        }

        // Wrap it into a VPIImage
        if (vimg_out == nullptr)
        {
            CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cvImageOut, VPI_IMAGE_FORMAT_Y8_ER, VPI_BACKEND_CUDA, &vimg_out));
        }
        // else
        // {
        //     CHECK_STATUS(vpiImageSetWrappedOpenCVMat(vimg_out, cvImage));
        // }

        CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, vimg, tmpIn, NULL));

        // Undistorts the input image.
        CHECK_STATUS(vpiSubmitRemap(stream, VPI_BACKEND_CUDA, remap, tmpIn, vimg_out, VPI_INTERP_CATMULL_ROM,
                                    VPI_BORDER_ZERO, 0));

        // Wait until conversion finishes.
        CHECK_STATUS(vpiStreamSync(stream));

        // Since vimg is wrapping the OpenCV image, the result is already there.
        // We just have to save it to disk.
    }

    uint16_t width_, height_;
    // VPI objects that will be used
    VPIStream stream = NULL;
    VPIPayload remap = NULL;
    VPIImage tmpIn = NULL;
    VPIImage vimg = nullptr, vimg_out = nullptr;
    
    VPIWarpMap map = {};
    VPIFisheyeLensDistortionModel distModel = {};
    VPICameraIntrinsic K;
    VPICameraExtrinsic X = {};

    bool do_not_rectify_ = false;
};

// class 