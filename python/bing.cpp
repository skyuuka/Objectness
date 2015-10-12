// Produce deprecation warnings (needs to come before arrayobject.h inclusion).
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <boost/python.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <memory>

#include "conversion.h"

#include "kyheader.h"
#include "Objectness.h"
#include "ValStructVec.h"

using namespace boost::python;

class Bing {

public:

    void load(std::string model_name) {
        DataSetVOC dummy;
        model_.reset(new Objectness(&dummy, 2, 8, 2));
        model_->loadTrainedModel(model_name);
    }

    PyObject* get_windows(PyObject *img_obj, int num_per_size = 130) {

        if (!model_) {
            throw std::runtime_error("model not loaded!");
        }

        NDArrayConverter cvt;
        cv::Mat img;
        img = cvt.toMat(img_obj);
        ValStructVec<float, cv::Vec4i> boxes;
        model_->getObjBndBoxes(img, boxes, num_per_size);
        cv::Mat b_mat = cv::Mat::zeros(boxes.size(), 4, CV_32F);

        for (int i = 0; i < boxes.size(); i++) {
            b_mat.at<float>(i, 0) = boxes[i][0]; // x1
            b_mat.at<float>(i, 1) = boxes[i][1]; // y1
            b_mat.at<float>(i, 2) = boxes[i][2]; // x2
            b_mat.at<float>(i, 3) = boxes[i][3]; // y2
        }

        return cvt.toNDArray(b_mat);
    }

private:
    std::shared_ptr<Objectness> model_;
};


BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(get_windows_overloads, get_windows, 1, 2)

BOOST_PYTHON_MODULE(cBing) 
{
    class_<Bing>("Bing", "Python Wrapper for Bing Objectness")
        .def("load", &Bing::load)
        .def("get_windows", &Bing::get_windows,
                get_windows_overloads(
                    args("img", "num_per_size"), 
                    "Get the region proposals of an image"
                    )
                )
    ;
}
