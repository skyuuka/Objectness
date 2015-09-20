#ifndef DATASETVOC_H
#define DATASETVOC_H

#include "DataSet.hpp"

class DataSetVOC: public DataSet
{
public:
	DataSetVOC() {};
	DataSetVOC(CStr &wkDir);
	~DataSetVOC(void);

	// Organization structure data for the dataset
	string wkDir; // Root working directory, all other directories are relative to this one
	string resDir, localDir; // Directory for saving results and local data
	string imgPathW, annoPathW; // Image and annotation path

	// Information for training and testing
	int trainNum, testNum;
	vecS trainSet, testSet; // File names (NE) for training and testing images
	vecS classNames; // Object class names
	vector<vector<Vec4i>> gtTrainBoxes, gtTestBoxes; // Ground truth bounding boxes for training and testing images
	vector<vecI> gtTrainClsIdx, gtTestClsIdx; // Object class indexes  


	// Load annotations
	void loadAnnotations();

	static bool cvt2OpenCVYml(CStr &annoDir); // Needs to call yml.m in this solution before running this function.

	// Get training and testing for demonstrating the generative of the objectness over classes
	void getTrainTest(); 

public: // Used for testing the ability of generic over classes
	void loadDataGenericOverCls();

private:
    void loadBox(const FileNode &fn, vector<Vec4i> &boxes, vecI &clsIdx);
	bool loadBBoxes(CStr &nameNE, vector<Vec4i> &boxes, vecI &clsIdx);
	static void getXmlStrVOC(CStr &fName, string &buf);
	static inline string keepXmlChar(CStr &str);
	static bool cvt2OpenCVYml(CStr &yamlName, CStr &ymlName); // Needs to call yml.m in this solution before running this function.
};

string DataSetVOC::keepXmlChar(CStr &_str)
{
	string str = _str;
	int sz = (int)str.size(), count = 0;
	for (int i = 0; i < sz; i++){
		char c = str[i];
		if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == ' ' || c == '.')
			str[count++] = str[i];
	}
	str.resize(count);
	return str;
}
#endif
