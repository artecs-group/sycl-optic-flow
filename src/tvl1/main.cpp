/* icpx -fsycl sycl_opencv_interop.cpp -o sycl_opencv_interop `pkg-config --cflags --libs opencv4`
 *
 * The example of interoperability between SYCL/OpenCL and OpenCV.
 * - SYCL: https://www.khronos.org/sycl/
 * - SYCL runtime parameters: https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md
 */
#include <string>
#include <CL/sycl.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;


class App
{
public:
    App(const CommandLineParser& cmd);
    void initVideoSource();
    void initSYCL();
    void process_frame(cv::Mat& frame);
    int run();
    bool isRunning() { return m_running; }
    bool doProcess() { return m_process; }
    void setRunning(bool running)      { m_running = running; }
    void setDoProcess(bool process)    { m_process = process; }
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

    cl::sycl::queue sycl_queue;
};


App::App(const CommandLineParser& cmd)
{
    m_camera_id  = cmd.get<int>("camera");
    m_file_name  = cmd.get<std::string>("video");

    m_running    = false;
    m_process    = false;
} // ctor


void App::initSYCL()
{
    using namespace cl::sycl;

    sycl_queue = cl::sycl::queue(cl::sycl::default_selector_v);

    auto device = sycl_queue.get_device();
    auto platform = device.get_platform();
    std::cout << "SYCL device: " << device.get_info<info::device::name>()
        << " @ " << device.get_info<info::device::driver_version>()
        << " (platform: " << platform.get_info<info::platform::name>() << ")" << std::endl;
} // initSYCL()


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
        m_cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        m_cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    }
    else
        throw std::runtime_error(std::string("specify video source"));
} // initVideoSource()


void App::process_frame(cv::Mat& frame)
{
    using namespace cl::sycl;

    // cv::Mat => cl::sycl::buffer
    {
        CV_Assert(frame.isContinuous());
        CV_CheckTypeEQ(frame.type(), CV_8UC1, "");

        buffer<uint8_t, 2> frame_buffer(frame.data, range<2>(frame.rows, frame.cols));

        sycl_queue.submit([&](handler& cgh) {
          auto pixels = frame_buffer.get_access<access::mode::read_write>(cgh);

          cgh.parallel_for<class sycl_inverse_kernel>(range<2>(frame.rows, frame.cols), [=](item<2> item) {
              uint8_t v = pixels[item];
              pixels[item] = ~v;
          });
        });

        sycl_queue.wait_and_throw();
    }

    {
        UMat blurResult;
        {
            UMat umat_buffer = frame.getUMat(ACCESS_RW);
            cv::blur(umat_buffer, blurResult, Size(3, 3));  // UMat doesn't support inplace
        }
        Mat result;
        blurResult.copyTo(result);
        swap(result, frame);
    }
}


int App::run() {
    std::cout << "Initializing..." << std::endl;

    initSYCL();
    initVideoSource();

    std::cout << "Press ESC to exit" << std::endl;
    std::cout << "      'p' to toggle ON/OFF processing" << std::endl;

    m_running = true;
    m_process = true;
    m_show_ui = true;

    std::string devName = sycl_queue.get_device().get_info<sycl::info::device::name>();

    int processedFrames = 0;

    cv::TickMeter timer;

    // Iterate over all frames
    while (isRunning() && m_cap.read(m_frame)) {
        timer.reset();
        timer.start();

        Mat m_frameGray;
        cvtColor(m_frame, m_frameGray, COLOR_BGR2GRAY);

        if (m_process)
            process_frame(m_frameGray);

        timer.stop();

        Mat img_to_show = m_frameGray;

        std::ostringstream msg, msg2;
        int currentFPS = 1000 / timer.getTimeMilli();
        msg << devName;
        msg2 << "FPS " << currentFPS << " (" << m_frame.size
            << ") Time: " << cv::format("%.2f", timer.getTimeMilli()) << " msec"
            << " (process: " << (m_process ? "True" : "False") << ")";

        cv::putText(img_to_show, msg.str(), Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 100, 0), 2);
        cv::putText(img_to_show, msg2.str(), Point(10, 50), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 100, 0), 2);

        if (m_show_ui) {
            try {
                imshow("Optic Flow", img_to_show);
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
                if (processedFrames > 0)
                    throw;
                m_show_ui = false;  // UI is not available
            }
        }

        processedFrames++;

        if (!m_show_ui && (processedFrames > 100)) 
            m_running = false;
    }

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
            return 1;
        }
        app.run();
    }
    catch (const cv::Exception& e)
    {
        std::cout << "FATAL: OpenCV error: " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception& e)
    {
        std::cout << "FATAL: C++ error: " << e.what() << std::endl;
        return 1;
    }

    catch (...)
    {
        std::cout << "FATAL: unknown C++ exception" << std::endl;
        return 1;
    }

    return EXIT_SUCCESS;
} // main()
