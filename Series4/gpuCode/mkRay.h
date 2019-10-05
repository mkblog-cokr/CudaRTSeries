#ifndef MKRAY_H
#define MKRAY_H
 
#include "mkVec3.h"
 
//MK: (코드 1- 1) GPU에서 호출가능 하도록 클래스 내부 함수에 __device__를 추가함
class ray{
    public:
        __device__ ray(){}
        __device__ ray(const vec3 &a, const vec3 &b){
            A = a;
            B = b;
        }

        __device__ vec3 origin() const{
            return A;
        }

        __device__ vec3 direction() const{
            return B;
        }

        __device__ vec3 pointAtParameter(float t) const{
            return (A + t * B);
        }

    private:
        vec3 A;
        vec3 B;
};

#endif
