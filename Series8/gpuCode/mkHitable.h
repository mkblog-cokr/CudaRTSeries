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
 
//MK: __device__를 추가해서 GPU에서 호출가능 하도록 함수를 변경함
class hitable{
    public: 
        __device__ virtual bool hit(const ray &r, float tMin, float tMax, hitRecord &rec) const = 0;
};
 
#endif
