#ifndef MKHITABLE_H
#define MKHITABLE_H
 
#include "mkVec3.h"

class material; 

//MK: Ray와 Hit한 Object의 위치를 파악하기 위해 사용
struct hitRecord{
    float t;
    vec3 p;
    vec3 normal;
	material *matPtr;
};
 
//MK: Hit여부를 판단하기 위한 추상클래스
class hitable{
    public: 
        virtual bool hit(const ray &r, float tMin, float tMax, hitRecord &rec) const = 0;
};
 
#endif
