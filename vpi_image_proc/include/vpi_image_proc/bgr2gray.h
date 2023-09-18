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

class VpiBGR2Gray{
public:
    VpiBGR2Gray(int width, int height){
        width_ = (int16_t)width;
        height_ = (int16_t)height;
        
        vpiStreamCreate(0, &stream);
        
        VPIConvertImageFormatParams cvtParams;
        vpiInitConvertImageFormatParams(&cvtParams);
        cvtParams.policy = VPI_CONVERSION_CLAMP;
    }
    ~VpiBGR2Gray(){
        vpiStreamDestroy(stream);
        vpiImageDestroy(input_img);
        vpiImageDestroy(output_img);
    }

    void compute(cv::Mat &cvImage, cv::Mat &cvImageOut){
        assert(!cvImage.empty());

        // Wrap it into a VPIImage
        if (input_img == nullptr)
        {
            CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cvImage, 0, &input_img));
        }
        else
        {
            CHECK_STATUS(vpiImageSetWrappedOpenCVMat(input_img, cvImage));
        }

        if (output_img == nullptr){
            CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cvImageOut, 0, &output_img));
        }
        else {
            CHECK_STATUS(vpiImageSetWrappedOpenCVMat(output_img, cvImageOut));
        }

        // Convert BGR -> GRAY
        CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, input_img, output_img, NULL));

        // Wait until conversion finishes.
        CHECK_STATUS(vpiStreamSync(stream));
    }

    uint16_t width_, height_;
    VPIStream stream = NULL;
    VPIImage input_img = NULL, output_img = NULL;
    VPIConvertImageFormatParams cvtParams;

};