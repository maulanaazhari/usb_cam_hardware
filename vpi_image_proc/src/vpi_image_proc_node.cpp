#include "vpi_image_proc/rectify.h"
#include "vpi_image_proc/bgr2gray.h"

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

#include <image_transport/image_transport.h>
#include <image_geometry/pinhole_camera_model.h>

class VPIImageProc{
public:
    VPIImageProc(ros::NodeHandle nh, ros::NodeHandle private_nh);
    ~VPIImageProc(){};

    void ImageCallback(const sensor_msgs::ImageConstPtr &image);
    void InfoCallback(const sensor_msgs::CameraInfoConstPtr &info);

private:
    ros::NodeHandle nh_, private_nh_;
    std::string image_topic_, cam_info_topic_, transport_hint_;
    // cv_bridge::CvImageConstPtr cv_image_;
    cv_bridge::CvImagePtr cv_image_, cv_image_out_;
    image_geometry::PinholeCameraModel camera_model_, camera_model_out_;
    std::shared_ptr<image_transport::ImageTransport> it_;
    image_transport::Subscriber image_subscriber_;
    image_transport::Publisher image_publisher_;
    image_transport::CameraPublisher image_cam_publisher_;
    // ros::Publisher compressed_image_out_pub_;
    ros::Publisher info_pub_;
    ros::Subscriber info_sub_;

    bool is_first_ = true;
    bool is_info_first_ = true;
    sensor_msgs::CameraInfoConstPtr cam_info_;
    sensor_msgs::CameraInfo cam_info_out_;
    sensor_msgs::ImagePtr img_ptr_;

    std::shared_ptr<VpiBGR2Gray> converter_;
    std::shared_ptr<VpiRectify> rectifier_;

};

VPIImageProc::VPIImageProc(ros::NodeHandle nh, ros::NodeHandle private_nh):
    nh_(nh), private_nh_(private_nh){
    private_nh_.param<std::string>("image_topic", image_topic_, "image");
    private_nh_.param<std::string>("camera_info_topic", cam_info_topic_, "camera_info");
    private_nh_.param<std::string>("transport_hint", transport_hint_, "compressed");

    // compressed_image_out_pub_ = nh_.advertise<sensor_msgs::CompressedImage>(ros::this_node::getNamespace() + "/image_rect_mono/compressed", 1);
    it_ = std::shared_ptr<image_transport::ImageTransport>(new image_transport::ImageTransport(private_nh_));
    image_subscriber_ = it_->subscribe(ros::this_node::getNamespace() + "/" + image_topic_, 1,
                                    &VPIImageProc::ImageCallback, this,
                                    image_transport::TransportHints(transport_hint_));
    info_sub_ = nh_.subscribe<sensor_msgs::CameraInfo>(ros::this_node::getNamespace() + "/" + cam_info_topic_, 1, &VPIImageProc::InfoCallback, this);
    image_publisher_ = it_->advertise("image_rect_mono", 1);
    info_pub_ = private_nh_.advertise<sensor_msgs::CameraInfo>("camera_info", 1);
    // it_->
}

void VPIImageProc::InfoCallback(const sensor_msgs::CameraInfoConstPtr &info){

    if(is_info_first_){
        cam_info_ = info;
        // image_geometry::PinholeCameraModel camera_model;
        camera_model_.fromCameraInfo(info);
        cam_info_out_ = sensor_msgs::CameraInfo();
        cam_info_out_.height = info->height;
        cam_info_out_.width = info->width;
        cam_info_out_.distortion_model = info->distortion_model;
        cam_info_out_.D = {0.0, 0.0, 0.0, 0.0};
        cam_info_out_.K = info->K;
        cam_info_out_.R = info->R;
        cam_info_out_.P = info->P;
        cam_info_out_.binning_x = info->binning_x;
        cam_info_out_.binning_y = info->binning_y;
        cam_info_out_.roi = info->roi;

        // Get camera intrinsic properties for rectified image.
        double fx = camera_model_.fx(); // focal length in camera x-direction [px]
        double fy = camera_model_.fy(); // focal length in camera y-direction [px]
        double cx = camera_model_.cx(); // optical center x-coordinate [px]
        double cy = camera_model_.cy(); // optical center y-coordinate [px]

        auto cameraMatrix = cv::Matx33d(fx, 0, cx,
                                0, fy, cy,
                                0, 0, 1);
        auto distCoeffs = camera_model_.distortionCoeffs();


        // converter_ = std::make_shared<VpiBGR2Gray>(info->width, info->height);
        rectifier_ = std::make_shared<VpiRectify>(info->width, info->height, cameraMatrix, distCoeffs);
    }

    info_sub_.shutdown();
        is_info_first_ = false;
}

void VPIImageProc::ImageCallback(const sensor_msgs::ImageConstPtr &image){
    if(is_info_first_) return;

    // if(compressed_image_out_pub_.getNumSubscribers() < 1) return;

    if(is_first_){
        cv_image_out_ = cv_bridge::toCvCopy(image, "mono8");
        is_first_ = false;
    }

    cv_image_ = cv_bridge::toCvCopy(image);

    // if(image->encoding != "mono8"){
            
        
    //     //convert the image
    //     converter_->compute(cv_image_->image, cv_image_out_->image);
    // }

    rectifier_->compute(cv_image_->image, cv_image_out_->image);

    img_ptr_ = cv_image_out_->toImageMsg();
    img_ptr_->header = cv_image_->header;
    cam_info_out_.header = cv_image_->header;
    image_publisher_.publish(img_ptr_);
    info_pub_.publish(cam_info_out_);
}


int main(int argc, char** argv) 
{
  ros::init(argc, argv, "vpi_image_proc");
  ros::NodeHandle nh, nhp("~");
  auto node = VPIImageProc(nh, nhp);
  ros::spin();
  
  return 0;
}