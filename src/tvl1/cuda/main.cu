#include <string>
#include <cmath>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include "tvl1.cuh"

using namespace cv;


class App {
public:
    App(const CommandLineParser& cmd);
    void initVideoSource();
    void initCuda();
    int run();
    bool isRunning() { return m_running; }
    bool doProcess() { return m_process; }
    void setRunning(bool running)      { m_running = running; }
    void setDoProcess(bool process)    { m_process = process; }
    void flowToColor(int width, int height, const float* flowData, cv::Mat& outFrame); 
protected:
    void handleKey(char key);
private:
    bool                        m_running;
    bool                        m_process;
    bool                        m_show_ui;

    int64_t                     m_t0;
    int64_t                     m_t1;
    float                       m_time;
    float                       m_frequency;

    std::string                 m_file_name;
    int                         m_camera_id;
    cv::VideoCapture            m_cap;
    cv::Mat                     m_frame;
};


App::App(const CommandLineParser& cmd)
{
    m_camera_id  = cmd.get<int>("camera");
    m_file_name  = cmd.get<std::string>("video");

    m_running    = false;
    m_process    = false;
} // ctor


void App::initCuda() {

}


void App::initVideoSource()
{
    if (!m_file_name.empty() && m_camera_id == -1)
    {
        m_cap.open(samples::findFileOrKeep(m_file_name));
        if (!m_cap.isOpened())
            throw std::runtime_error(std::string("can't open video stream: ") + m_file_name);
    }
    else if (m_camera_id != -1)
    {
        m_cap.open(m_camera_id);
        if (!m_cap.isOpened())
            throw std::runtime_error(std::string("can't open camera: ") + std::to_string(m_camera_id));
        // m_cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        // m_cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    }
    else
        throw std::runtime_error(std::string("specify video source"));
} // initVideoSource()


void App::flowToColor(int width, int height, const float* flowData, cv::Mat& outFrame) {
    cv::Mat flow_magnitude(height, width, CV_32F);
    cv::Mat flow_angle(height, width, CV_32F);

    // Populate the flow_magnitude and flow_angle matrices from the flowData array
    #pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        #pragma omp simd
        for (int x = 0; x < width; ++x) {
            int index = y * width + x;
            float u = flowData[index];
            float v = flowData[index + width * height];
            flow_magnitude.at<float>(y, x) = std::sqrt(u * u + v * v);
            flow_angle.at<float>(y, x) = std::atan2(v, u);
        }
    }

    flow_magnitude = cv::min(flow_magnitude * 10.0f, 255.0f);
    flow_magnitude.convertTo(flow_magnitude, CV_8U);

    flow_angle = flow_angle * 180.0f / CV_PI / 2.0f;
    flow_angle.convertTo(flow_angle, CV_8U);

    cv::Mat hsv(height, width, CV_8UC3, cv::Scalar(0, 255, 255));

    hsv.at<cv::Vec3b>(cv::Point(0, 0))[0] = 0;
    #pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        #pragma omp simd
        for (int x = 0; x < width; ++x) {
            hsv.at<cv::Vec3b>(y, x)[0] = static_cast<uint8_t>(flow_angle.at<uint8_t>(y, x));
            hsv.at<cv::Vec3b>(y, x)[1] = 255;
            hsv.at<cv::Vec3b>(y, x)[2] = flow_magnitude.at<uint8_t>(y, x);
        }
    }

    cv::cvtColor(hsv, outFrame, cv::COLOR_HSV2BGR);
}


int App::run() {
    std::cout << "Initializing..." << std::endl;

    //initCuda();
    cudaDeviceProp devProps;
    cudaGetDeviceProperties(&devProps, 0);
    std::string devName = devProps.name; 
    
    initVideoSource();

    std::cout << "Press ESC to exit" << std::endl;
    std::cout << "      'p' to toggle ON/OFF processing" << std::endl;

    m_running = true;
    m_process = true;
    m_show_ui = true;

    const int width  = static_cast<int>(m_cap.get(cv::CAP_PROP_FRAME_WIDTH));
	const int height = static_cast<int>(m_cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // buffers required for the image proccesing
    float* img = new float[width * height]{0};
    TV_L1 tvl1 = TV_L1(width, height);
    unsigned int processedFrames{0};
    double fps{0};
    cv::TickMeter timer;
    
    // Iterate over all frames
    try {
        while (isRunning()) {
            timer.reset();
            timer.start();

            m_cap.read(m_frame);

            cv::Mat m_frameGray;
            cv::cvtColor(m_frame, m_frameGray, COLOR_BGR2GRAY);

            if (m_process) {
                #pragma omp parallel for simd
                for (size_t i = 0; i < width*height; i++) {
                    img[i] = static_cast<float>(m_frameGray.data[i]);
                }
                try {
                    tvl1.runDualTVL1Multiscale(img);
                    flowToColor(width, height, tvl1.getU(), m_frameGray);
                }
                catch(const std::exception& e) {
                    std::cerr << e.what() << '\n';
                }
            }
            timer.stop();

            cv::Mat imgToShow = m_frameGray;

            std::ostringstream msg, msg2;
            fps += 1000 / timer.getTimeMilli();
            int currentFPS = 1000 / timer.getTimeMilli();
            msg << devName;
            msg2 << "FPS " << currentFPS << " (" << imgToShow.size
                << ") Time: " << cv::format("%.2f", timer.getTimeMilli()) << " msec"
                << " (process: " << (m_process ? "True" : "False") << ")";

            cv::putText(imgToShow, msg.str(), Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 100, 0), 2);
            cv::putText(imgToShow, msg2.str(), Point(10, 50), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 100, 0), 2);

            if (m_show_ui) {
                try {
                    cv::imshow("Optic Flow", imgToShow);
                    int key = waitKey(1);
                    switch (key) {
                    case 27:  // ESC
                        m_running = false;
                        break;

                    case 'p':  // fallthru
                    case 'P':
                        m_process = !m_process;
                        break;

                    default:
                        break;
                    }
                }
                catch (const std::exception& e) {
                    std::cerr << "ERROR(OpenCV UI): " << e.what() << std::endl;
                    if (processedFrames > 0) {
                        delete[] img;
                        throw;
                    }
                    m_show_ui = false;  // UI is not available
                }
            }

            processedFrames++;

            if (!m_show_ui && (processedFrames > 100)) 
                m_running = false;
        }
        std::cout << std::endl;
        std::cout << "Number of frames = " << processedFrames << std::endl;
        std::cout << "Avg of FPS = " << cv::format("%.2f", fps / processedFrames) << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << std::endl;
        std::cout << "Number of frames = " << processedFrames << std::endl;
        std::cout << "Avg of FPS = " << cv::format("%.2f", fps / processedFrames) << std::endl;
        delete[] img;
        return 0;
    }

    delete[] img;
    return 0;
}


int main(int argc, char** argv)
{
    const char* keys =
        "{ help h ?    |          | print help message }"
        "{ camera c    | -1       | use camera as input }"
        "{ video  v    |          | use video as input }";

    CommandLineParser cmd(argc, argv, keys);
    if (cmd.has("help"))
    {
        cmd.printMessage();
        return EXIT_SUCCESS;
    }

    try
    {
        App app(cmd);
        if (!cmd.check())
        {
            cmd.printErrors();
            return EXIT_FAILURE;
        }
        app.run();
    }
    catch (const cv::Exception& e)
    {
        std::cout << "FATAL: OpenCV error: " << e.what() << std::endl;
        return EXIT_SUCCESS;
    }
    catch (const std::exception& e)
    {
        std::cout << "FATAL: C++ error: " << e.what() << std::endl;
        return EXIT_SUCCESS;
    }

    catch (...)
    {
        std::cout << "FATAL: unknown C++ exception" << std::endl;
        return EXIT_SUCCESS;
    }

    return EXIT_SUCCESS;
} // main()
