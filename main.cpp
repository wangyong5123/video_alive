#include<opencv2\opencv.hpp>
#include<opencv2\video\video.hpp>

extern "C"
{
#include "libswscale/swscale.h"
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
}

#pragma comment(lib, "avcodec.lib")
#pragma comment(lib, "avutil.lib")
#pragma comment(lib, "avformat.lib")
#pragma comment(lib, "swscale.lib")
using namespace std;
using namespace cv;

using namespace cv::dnn;

const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const float confidenceThreshold = 0.7;
const cv::Scalar meanVal(104.0, 177.0, 123.0);

#define CAFFE

const std::string caffeConfigFile = "./models/deploy.prototxt";
const std::string caffeWeightFile = "./models/res10_300x300_ssd_iter_140000_fp16.caffemodel";

const std::string tensorflowConfigFile = "./models/opencv_face_detector.pbtxt";
const std::string tensorflowWeightFile = "./models/opencv_face_detector_uint8.pb";

void detectFaceOpenCVDNN(Net net, Mat &frameOpenCVDNN)
{
	int frameHeight = frameOpenCVDNN.rows;
	int frameWidth = frameOpenCVDNN.cols;
#ifdef CAFFE
	cv::Mat inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, false, false);
#else
	cv::Mat inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, true, false);
#endif

	net.setInput(inputBlob, "data");
	cv::Mat detection = net.forward("detection_out");

	cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

	for (int i = 0; i < detectionMat.rows; i++)
	{
		float confidence = detectionMat.at<float>(i, 2);

		if (confidence > confidenceThreshold)
		{
			int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
			int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
			int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
			int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);

			cv::rectangle(frameOpenCVDNN, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2, 4);
		}
	}

}

int main()
{
	//nginx-rtmp ֱ��������rtmp����URL
	const char *outUrl = "rtmp://192.168.11.102/live/1";

	//ע�����еı������
	avcodec_register_all();

	//ע�����еķ�װ��
	av_register_all();

	//ע����������Э��
	avformat_network_init();
	VideoCapture cam(0);
	Mat frame;

	//���ظ�ʽת��������
	SwsContext *vsc = NULL;

	//��������ݽṹ
	AVFrame *yuv = NULL;

	//������������
	AVCodecContext *vc = NULL;

	//rtmp flv ��װ��
	AVFormatContext *ic = NULL;

#ifdef CAFFE
	Net net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);
#else
	Net net = cv::dnn::readNetFromTensorflow(tensorflowWeightFile, tensorflowConfigFile);
#endif

	try
	{	////////////////////////////////////////////////////////////////
		/// 1 ʹ��opencv��rtsp��� cam.open(inUrl)
		if (!cam.isOpened())
		{
			throw exception("cam open failed!");
		}
		//cout << inUrl << " cam open success" << endl;
		int inWidth = cam.get(CAP_PROP_FRAME_WIDTH);
		int inHeight = cam.get(CAP_PROP_FRAME_HEIGHT);
		int fps = cam.get(CAP_PROP_FPS);

		///2 ��ʼ����ʽת��������
		vsc = sws_getCachedContext(vsc,
			inWidth, inHeight, AV_PIX_FMT_BGR24,	 //Դ���ߡ����ظ�ʽ
			inWidth, inHeight, AV_PIX_FMT_YUV420P,//Ŀ����ߡ����ظ�ʽ
			SWS_BICUBIC,  // �ߴ�仯ʹ���㷨
			0, 0, 0
		);
		if (!vsc)
		{
			throw exception("sws_getCachedContext failed!");
		}
		///3 ��ʼ����������ݽṹ
		yuv = av_frame_alloc();
		yuv->format = AV_PIX_FMT_YUV420P;
		yuv->width = inWidth;
		yuv->height = inHeight;
		yuv->pts = 0;
		//����yuv�ռ�
		int ret = av_frame_get_buffer(yuv, 32);
		if (ret != 0)
		{
			char buf[1024] = { 0 };
			av_strerror(ret, buf, sizeof(buf) - 1);
			throw exception(buf);
		}

		///4 ��ʼ������������
		//a �ҵ�������
		AVCodec *codec = avcodec_find_encoder(AV_CODEC_ID_H264);
		if (!codec)
		{
			throw exception("Can`t find h264 encoder!");
		}
		//b ����������������
		vc = avcodec_alloc_context3(codec);
		if (!vc)
		{
			throw exception("avcodec_alloc_context3 failed!");
		}
		//c ���ñ���������
		vc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER; //ȫ�ֲ���
		vc->codec_id = codec->id;
		vc->thread_count = 8;

		vc->bit_rate = 50 * 1024 * 8;//ѹ����ÿ����Ƶ��bitλ��С 50kB
		vc->width = inWidth;
		vc->height = inHeight;
		vc->time_base = { 1,fps };
		vc->framerate = { fps,1 };

		//������Ĵ�С������֡һ���ؼ�֡
		vc->gop_size = 50;
		vc->max_b_frames = 0;
		vc->pix_fmt = AV_PIX_FMT_YUV420P;
		//d �򿪱�����������
		ret = avcodec_open2(vc, 0, 0);
		if (ret != 0)
		{
			char buf[1024] = { 0 };
			av_strerror(ret, buf, sizeof(buf) - 1);
			throw exception(buf);
		}
		cout << "avcodec_open2 success!" << endl;

		///5 �����װ������Ƶ������
		//a ���������װ��������
		ret = avformat_alloc_output_context2(&ic, 0, "flv", outUrl);
		if (ret != 0)
		{
			char buf[1024] = { 0 };
			av_strerror(ret, buf, sizeof(buf) - 1);
			throw exception(buf);
		}
		//b �����Ƶ�� 
		AVStream *vs = avformat_new_stream(ic, NULL);
		if (!vs)
		{
			throw exception("avformat_new_stream failed");
		}
		//vs->codec->codec_tag = 0;
		//�ӱ��������Ʋ���
		avcodec_parameters_from_context(vs->codecpar, vc);
		av_dump_format(ic, 0, outUrl, 1);


		///��rtmp ���������IO
		ret = avio_open(&ic->pb, outUrl, AVIO_FLAG_WRITE);
		if (ret != 0)
		{
			char buf[1024] = { 0 };
			av_strerror(ret, buf, sizeof(buf) - 1);
			throw exception(buf);
		}

		//д���װͷ
		ret = avformat_write_header(ic, NULL);
		if (ret != 0)
		{
			char buf[1024] = { 0 };
			av_strerror(ret, buf, sizeof(buf) - 1);
			throw exception(buf);
		}

		AVPacket pack;
		memset(&pack, 0, sizeof(pack));
		int vpts = 0;
		double tt_opencvDNN = 0;
		double fpsOpencvDNN = 0;
		for (;;)
		{
			///��ȡrtsp��Ƶ֡��������Ƶ֡
			if (!cam.grab())
			{
				continue;
			}
			///yuvת��Ϊrgb
			if (!cam.retrieve(frame))
			{
				continue;
			}

			if (frame.empty())
				break;
			double t = cv::getTickCount();
			//����ʶ���㷨
			detectFaceOpenCVDNN(net, frame);
			tt_opencvDNN = ((double)cv::getTickCount() - t)  / cv::getTickFrequency();
			fpsOpencvDNN = 1 / tt_opencvDNN;
			putText(frame, format("OpenCV DNN ; FPS = %.2f", tt_opencvDNN), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.4, Scalar(0, 0, 255), 4);

			///rgb to yuv
			uint8_t *indata[AV_NUM_DATA_POINTERS] = { 0 };
			//indata[0] bgrbgrbgr
			//plane indata[0] bbbbb indata[1]ggggg indata[2]rrrrr 
			indata[0] = frame.data;
			int insize[AV_NUM_DATA_POINTERS] = { 0 };
			//һ�У������ݵ��ֽ���
			insize[0] = frame.cols * frame.elemSize();
			int h = sws_scale(vsc, indata, insize, 0, frame.rows, //Դ����
				yuv->data, yuv->linesize);
			if (h <= 0)
			{
				continue;
			}
			//cout << h << " " << flush;
			///h264����
			yuv->pts = vpts;
			vpts++;
			ret = avcodec_send_frame(vc, yuv);
			if (ret != 0)
				continue;

			ret = avcodec_receive_packet(vc, &pack);
			if (ret != 0 || pack.size > 0)
			{
				//cout << "*" << pack.size << flush;
			}
			else
			{
				continue;
			}
			//����
			pack.pts = av_rescale_q(pack.pts, vc->time_base, vs->time_base);
			pack.dts = av_rescale_q(pack.dts, vc->time_base, vs->time_base);
			pack.duration = av_rescale_q(pack.duration, vc->time_base, vs->time_base);
			ret = av_interleaved_write_frame(ic, &pack);

			
			if (ret == 0)
			{
				cout << "#" << flush;
			}
		}

	}
	catch (exception &ex)
	{
		if (cam.isOpened())
			cam.release();
		if (vsc)
		{
			sws_freeContext(vsc);
			vsc = NULL;
		}

		if (vc)
		{
			avio_closep(&ic->pb);
			avcodec_free_context(&vc);
		}

		cerr << ex.what() << endl;
	}
	getchar();
	return 0; 

	return 2;
}


