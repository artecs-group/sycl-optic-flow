#include "opencv2/opencv.hpp"
#include <iostream>
#include <math.h>
#include "lucaskanade.hpp"

using namespace std;
using namespace cv;

#define MY_PI 3.14159

#define VIDEO_

// Copy frame into wind_frs
void copy_frames_circularbuffer(unsigned char *frame, unsigned char *wind_frs, int temp_size, int iframe, int frame_size)
{
	int idx = iframe%temp_size;

	memcpy( wind_frs + idx*frame_size, frame, frame_size );
}



void post_video(uchar* video_frames, float* Vx, float*Vy, int nframes, int nx, int ny){
	double fps = 4.0;
	unsigned char *ang = new unsigned char[nx*ny];
	unsigned char *vel = new unsigned char[nx*ny];
	unsigned char *data_out = new unsigned char[3*(ny+10)*nx];

	
	for(int i=0; i<3*(ny+10)*nx; i++)
		data_out[i] = 0;
	
	int size_frame = nx*ny;
	
	float velmax = 4.f;
	
	int codec = cv::VideoWriter::fourcc('F','F', 'V','1'); // select desired codec (must be available at runtime)
	
	cv::VideoWriter video_out;
	video_out.open("flow.avi", codec, fps, cv::Size(nx, 3*(ny+10)), false);
	
	for (int fr=0; fr<nframes; fr++){
		for (int i=0; i<ny; i++)
			for (int j=0; j<nx; j++){
				float vx = Vx[fr*size_frame + i*nx+j];
				float vy = Vy[fr*size_frame + i*nx+j];
				vel[i*nx+j] = sqrtf(vx*vx+vy*vy)/velmax*255.f;                     //convert to 0-255
				ang[i*nx+j] = ((atan2f(vy, vx)*360.f/2/MY_PI + 180.f)/360.f)*255.f; //atan2=(-pi,pi) convert to 0-255
			}

		memcpy( data_out               , video_frames+fr*ny*nx, ny*nx ); //Copy input video
		memcpy( data_out +   (ny+10)*nx, vel                  , ny*nx ); //Copy vel video
		memcpy( data_out + 2*(ny+10)*nx, ang                  , ny*nx ); //Copy ang video
		
		cv::Mat image_outMat = cv::Mat(3*(ny+10), nx, CV_8UC1, data_out);
	        video_out.write(image_outMat);
	}

	video_out.release();
	
	delete[] ang;
	delete[] vel;
	delete[] data_out;
}

int main(){

	int gpu_enable = 1;
	int temp_conv_size=2;
	int spac_conv_size=3; 
	int window_size=5;
	int nx = 150, ny = 150, nframes = 40;
	char file_name[30];
#ifdef VIDEO
	strcpy(file_name, "dataset/translatingtree_150x150_40frs_8fps.mp4");
#endif
	uchar* wind_frs = new unsigned char [temp_conv_size*nx*ny];
	
	uchar* video_frames = new unsigned char[nframes*nx*ny];
	
	float *Vx = new float [nframes*nx*ny];
	float *Vy = new float [nframes*nx*ny];

	setbuf(stdout,NULL);

	// Init memory and filters */
	LucasKanade lk(spac_conv_size, temp_conv_size, window_size, nx, ny, gpu_enable);

#ifdef VIDEO
	// Create a VideoCapture object and open the input file
	// If the input is the web camera, pass 0 instead of the video file name
	VideoCapture cap(file_name); 

	// Check if camera opened successfully
	if(!cap.isOpened()){
		cout << "Error opening video stream or file" << endl;
		return -1;
	}
#endif

	for(int fr=0; fr<nframes; fr++)
	{
		Mat frame;
		uchar* raw_frame;
#ifdef VIDEO
		// Capture frame-by-frame
		cap >> frame;
		if (frame.isContinuous())
			raw_frame = frame.data;
		else
			printf("asdfaf@\n");
#else
		char image_fname[40];
		sprintf(image_fname, "dataset/translatingtree_%d.png", fr);		
		frame = cv::imread(image_fname, cv::IMREAD_GRAYSCALE);
		raw_frame = frame.data;
#endif

		memcpy( video_frames+fr*ny*nx, raw_frame, ny*nx );
		copy_frames_circularbuffer(raw_frame, wind_frs, temp_conv_size, fr, nx*ny);
		if (gpu_enable)
			lk.copy_frames_circularbuffer_GPU_wrapper(raw_frame, temp_conv_size, fr, nx*ny);

		// If the frame is empty, break immediately
		if (frame.empty())
			break;

		if (fr>=temp_conv_size-1){
			lk.lucas_kanade(Vx+fr*nx*ny, Vy+fr*nx*ny, fr, wind_frs, gpu_enable); //GPU enable: 1
		}

		// Display the resulting frame
		//imshow( "Frame", frame );

		// Press  ESC on keyboard to exit
		char c=(char)waitKey(25);
		if(c==27)
			break;
	}

#ifdef VIDEO
	// When everything done, release the video capture object
	cap.release();
#endif
	// Closes all the frames
	destroyAllWindows();
	printf("Done!!!!\n");
	post_video(video_frames, Vx, Vy, nframes, nx, ny);
	
	delete[] video_frames;
	delete[] Vx;
	delete[] Vy;

	return 0;
}

