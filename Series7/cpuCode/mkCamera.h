#ifndef MKCAMERA_H
#define MKCAMERA_H
 
#include "mkRay.h"
 
//MK: 간단한 Camera 클래스 
//MK: 이번 장에서는 단순히 Ray만 생성해주는 코드만 포함하고 있음
class camera{
    public:
        camera(){
            lowerLeftCorner = vec3(-2.0, -1.0, -1.0);
            horizontal = vec3(4.0, 0.0, 0.0);
            vertical = vec3(0.0, 2.0, 0.0);
            origin = vec3(0.0, 0.0, 0.0);
        }
        ray getRay(float u, float v){
            return ray(origin, lowerLeftCorner + u * horizontal + v * vertical - origin);
        }
 
    private:
        vec3 origin;
        vec3 lowerLeftCorner;
        vec3 horizontal;
        vec3 vertical;
};
 
#endif
