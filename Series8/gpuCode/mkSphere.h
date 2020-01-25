#ifndef MKSPHERE_H
#define MKSPHERE_H
 
#include "mkHitable.h"
#include "mkMaterial.h"
 
class sphere: public hitable{
    public:
        __device__ sphere(){}
        __device__ sphere(vec3 cen, float r, material *ptr) : center(cen), radius(r), matPtr(ptr) {}
 
        __device__ virtual bool hit(const ray &r, float tMin, float tMax, hitRecord &rec) const {
            vec3 oc = r.origin() - center;
            float a = dot(r.direction(), r.direction());
            float b = dot(oc, r.direction());
            float c = dot(oc, oc) - radius * radius;
            float discriminant = b*b - a*c;
            if(discriminant > 0){
                float temp = (-b - sqrt(discriminant))/a;
                //MK: 구의 가까운 부분 부터 Hit여부를 판단함
                if(temp < tMax && temp > tMin){
                    rec.t = temp;
                    rec.p = r.pointAtParameter(rec.t);
                    rec.normal = (rec.p - center) / radius;
					//MK: (코드 3) 구의 Material 정보를 저장함
					rec.matPtr = matPtr;
                    return true;
                }
                temp = (-b + sqrt(discriminant))/a;
                if(temp < tMax && temp > tMin){
                    rec.t = temp;
                    rec.p = r.pointAtParameter(rec.t);
                    rec.normal = (rec.p - center) / radius;
					//MK: (코드 3) 구의 Material 정보를 저장함
                    rec.matPtr = matPtr;
					return true;
                }
            }
            return false;
        }
 
        vec3 center;
        float radius;
		material *matPtr;
};
 
#endif
