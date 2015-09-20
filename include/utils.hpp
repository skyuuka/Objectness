#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>


inline double bbox_overlap(const Vec4i &bb1, const Vec4i &bb2) 
{
	int bi[4];
	bi[0] = max(bb1[0], bb2[0]);
	bi[1] = max(bb1[1], bb2[1]);
	bi[2] = min(bb1[2], bb2[2]);
	bi[3] = min(bb1[3], bb2[3]);	

	double iw = bi[2] - bi[0] + 1;
	double ih = bi[3] - bi[1] + 1;
	double ov = 0;
	if (iw>0 && ih>0){
		double ua = (bb1[2]-bb1[0]+1)*(bb1[3]-bb1[1]+1)+(bb2[2]-bb2[0]+1)*(bb2[3]-bb2[1]+1)-iw*ih;
		ov = iw*ih/ua;
	}	
	return ov;
}
#endif
