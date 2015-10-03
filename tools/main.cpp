#include "kyheader.h"
#include "Objectness.h"
#include "ValStructVec.h"
#include "CmShow.h"

void RunObjectness(CStr &resName, double base, int W, int NSS, int numPerSz);

void illutrateLoG()
{
    for (float delta = 0.5f; delta < 1.1f; delta+=0.1f){
        Mat f = Objectness::aFilter(delta, 8);
        normalize(f, f, 0, 1, NORM_MINMAX);
        CmShow::showTinyMat(format("D=%g", delta), f);
    }
    waitKey(0);
}


inline bool check_exists(const std::string & name) {
    struct stat buffer;
    return (stat (name.c_str() , &buffer) == 0);
}


void writeBoxesToFile(std::string const& textFileName, ValStructVec<float, Vec4i> const& bboxes)
{
    std::ofstream f(textFileName.c_str());
    if (f.is_open() == false) {
        std::cerr << "Error: failed to open " << textFileName << std::endl;
    }
    f << bboxes.size() << std::endl;
    for (int j = 0; j < bboxes.size(); j++) {
        f << bboxes[j][0] << " " << bboxes[j][1] << " " << bboxes[j][2] << " " << bboxes[j][3] << std::endl;
    }
    f.close();
}


void print_help(const string &program_name) 
{
    printf("Usage: %s baseImageDir -f imageNameList -w\n", program_name.c_str());
    exit(1);
}

void RunCustomerImages(int argc, char* argv[])
{
    string baseImageDir; 
    string imageNameListFullPath; 
    bool overwrite = false;
    if (argc < 2 || argc > 5) {
        print_help(argv[0]);
    } else {
        baseImageDir = argv[1];
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "-f") == 0) {
                if (i + 1 != argc) {
                    imageNameListFullPath = argv[i + 1];
                } else {
                    print_help(argv[0]);
                }
            } else if (strcmp(argv[i], "-w") == 0) {
                overwrite = true;
            }
        } 
    }

    if (imageNameListFullPath.size() == 0) {
        imageNameListFullPath = baseImageDir + "/" + "imageNameList.txt";
    }

    cout << "baseImageDir = " << baseImageDir << endl;
    cout << "imageNameListFullPath = " << imageNameListFullPath << endl;
    cout << "overwrite = " << overwrite << endl;

    srand((unsigned int)time(NULL));
    DataSetVOC voc2007("/home/lin/data/VOCdevkit2007/VOC2007/");
    voc2007.loadAnnotations();
    double base = 2;
    int W = 8;
    int NSS = 2;
    int numPerSz = 100;
    printf("Dataset:`%s' with %d training and %d testing\n", _S(voc2007.wkDir), voc2007.trainNum, voc2007.testNum);
    printf("Base = %g, W = %d, NSS = %d, perSz = %d\n", base, W, NSS, numPerSz);
    Objectness objNess(&voc2007, base, W, NSS);
    //objNess.trainObjectness(numPerSz);
    objNess.loadTrainedModel();


    CmTimer tm("single");
    float total = 0.;
    vecS imageFileNames = CmFile::loadStrList(imageNameListFullPath);
    for (int i = 0; i < imageFileNames.size(); i++) {
        CStr fileNameFullPath = baseImageDir + "/" + imageFileNames[i];
        unsigned int fileNameEndLoc = fileNameFullPath.find_last_of('.');
        string outFileName = fileNameFullPath.substr(0, fileNameEndLoc) + "_bing_voc.txt";
        if (check_exists(outFileName) && ! overwrite) {
            continue;
        }
        //Mat img = imread("/workplace/linche/Logo/data/CamFind-Logo/converse/6_6895_6895645_image.jpg");
        Mat img = imread(fileNameFullPath);
        ValStructVec<float, Vec4i> bboxes;
        tm.Start();
        objNess.getObjBndBoxes(img, bboxes, numPerSz);
        tm.Stop();
        total += tm.TimeInSeconds();
        clock_t end_time = clock();
        cout << "[" << i << "/" << imageFileNames.size() << "]" << fileNameFullPath << ":" << bboxes.size() << " regions detected using " << tm.TimeInSeconds() << "s" << endl;
        writeBoxesToFile(outFileName, bboxes);
        /*
        for (int j = 0; j < bboxes.size(); j++) {
            Rect current(Point2i(bboxes[j][0], bboxes[j][1]), Point2i(bboxes[j][2], bboxes[j][3]));
            rectangle(img, current, Scalar(255, 0, 0), 1);
            imshow("query", img);
            waitKey(0);
        }
        */
    }
    cout << "average time per image = " << total / imageFileNames.size() << "s" << endl;
}

int main(int argc, char* argv[])
{
    //CStr wkDir = "D:/WkDir/DetectionProposals/VOC2007/Local/";
    //illutrateLoG();
    //RunObjectness("WinRecall.m", 2, 8, 2, 130);
    RunCustomerImages(argc, argv);
    return 0;
}

void RunObjectness(CStr &resName, double base, int W, int NSS, int numPerSz)
{
    srand((unsigned int)time(NULL));
    //DataSetVOC voc2007("/home/bittnt/BING/BING_beta1/VOC/VOC2007/");
    DataSetVOC voc2007("/home/lin/data/VOCdevkit2007/VOC2007/");
    voc2007.loadAnnotations();
    //voc2007.loadDataGenericOverCls();

    printf("Dataset:`%s' with %d training and %d testing\n", _S(voc2007.wkDir), voc2007.trainNum, voc2007.testNum);
    printf("%s Base = %g, W = %d, NSS = %d, perSz = %d\n", _S(resName), base, W, NSS, numPerSz);

    Objectness objNess(&voc2007, base, W, NSS);

    vector<vector<Vec4i> > boxesTests;
    //objNess.getObjBndBoxesForTests(boxesTests, 250);
    objNess.getObjBndBoxesForTestsFast(boxesTests, numPerSz);
    //objNess.getRandomBoxes(boxesTests);

    //objNess.evaluatePerClassRecall(boxesTests, resName, 1000);
    //objNess.illuTestReults(boxesTests);
    //objNess.evaluatePAMI12();
    //objNess.evaluateIJCV13();
}
