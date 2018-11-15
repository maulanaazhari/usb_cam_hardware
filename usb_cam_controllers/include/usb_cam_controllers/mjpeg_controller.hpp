#ifndef USB_CAM_CONTROLLERS_MJPEG_CONTROLLER
#define USB_CAM_CONTROLLERS_MJPEG_CONTROLLER

#include <string>
#include <vector>

#include <controller_interface/controller.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <image_transport/publisher.h>
#include <ros/console.h>
#include <ros/duration.h>
#include <ros/node_handle.h>
#include <ros/time.h>
#include <usb_cam_hardware_interface/packet_interface.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
}

namespace usb_cam_controllers {

class MjpegController
    : public controller_interface::Controller< usb_cam_hardware_interface::PacketInterface > {
public:
  MjpegController() {}

  virtual ~MjpegController() {
    avcodec_free_context(&av_context_);
    av_frame_free(&av_frame_);
    av_packet_free(&av_packet_);
  }

  virtual bool init(usb_cam_hardware_interface::PacketInterface *hw, ros::NodeHandle &root_nh,
                    ros::NodeHandle &controller_nh) {
    //
    // load parameters
    //

    encoding_ = controller_nh.param< std::string >("encoding", "bgr8");

    //
    // get hardware handle
    //

    if (!hw) {
      ROS_ERROR("Null packet interface");
      return false;
    }

    const std::vector< std::string > names(hw->getNames());
    if (names.empty()) {
      ROS_ERROR("No packet handle");
      return false;
    }
    if (names.size() > 1) {
      ROS_WARN_STREAM(names.size()
                      << " packet handles. packets only from the first handle will be published.");
    }

    packet_ = hw->getHandle(names.front());

    //
    // setup mjpeg decoder
    //

    avcodec_register_all();
    av_decoder_ = avcodec_find_decoder(AV_CODEC_ID_MJPEG);
    if (!av_decoder_) {
      ROS_ERROR("Cannot find mjpeg decoder");
      return false;
    }
    av_context_ = avcodec_alloc_context3(av_decoder_);
    if (avcodec_open2(av_context_, av_decoder_, 0) < 0) {
      ROS_ERROR("Cannot open mjpeg decoder");
      return false;
    }
    av_packet_ = av_packet_alloc();
    av_frame_ = av_frame_alloc();

    //
    // setup image publisher
    //

    publisher_ = image_transport::ImageTransport(controller_nh).advertise("image", 1);
    last_stamp_ = ros::Time(0);

    return true;
  }

  virtual void starting(const ros::Time &time) {
    // nothig to do
  }

  virtual void update(const ros::Time &time, const ros::Duration &period) {
    // validate the packet
    if (!packet_.getStart()) {
      ROS_INFO("No packet to be published");
      return;
    }
    if (packet_.getStamp() == last_stamp_) {
      ROS_INFO("Packet has already been published");
      return;
    }

    // decode the packet and get a frame
    av_packet_->data = const_cast< uint8_t * >(packet_.getStartAs< uint8_t >());
    av_packet_->size = packet_.getLength();
    if (avcodec_send_packet(av_context_, av_packet_) < 0) {
      ROS_ERROR("Cannot send packet to decoder");
      return;
    }
    if (avcodec_receive_frame(av_context_, av_frame_) < 0) {
      ROS_ERROR("Cannot receive frame from decoder");
      return;
    }

    // TODO: convert color spaces of the frame

    // publish the frame
    cv_bridge::CvImage image;
    image.header.stamp = packet_.getStamp();
    image.encoding = encoding_;
    // image.image = ;
    publisher_.publish(image.toImageMsg());
    last_stamp_ = packet_.getStamp();
  }

  virtual void stopping(const ros::Time &time) {
    // nothing to do
  }

private:
  std::string encoding_;

  AVCodec *av_decoder_;
  AVCodecContext *av_context_;
  AVFrame *av_frame_;
  AVPacket *av_packet_;

  usb_cam_hardware_interface::PacketHandle packet_;
  image_transport::Publisher publisher_;
  ros::Time last_stamp_;
};

} // namespace usb_cam_controllers

#endif