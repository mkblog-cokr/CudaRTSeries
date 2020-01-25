#ifndef MKMATERIAL_H
#define MKMATERIAL_H

#include "mkVec3.h"
#include "mkRay.h"

struct hitRecord;

//MK: (코드 1) Material Class를 새로 추가함

class material{
    public:
        __device__ virtual bool scatter(const ray &rIn, const hitRecord &rec, vec3 &attenuation, ray &scattered, curandState *localRandState) const = 0;
};

class lambertian : public material{
    public:
        __device__ lambertian(const vec3 &a): albedo(a) {}
        __device__ virtual bool scatter(const ray &rIn, const hitRecord &rec, vec3 &attenuation, ray &scattered, curandState *localRandState) const {
            vec3 target = rec.p + rec.normal + randomInUnitSphere(localRandState);
			scattered = ray(rec.p, unitVector(target-rec.p));
            attenuation = albedo;
            return true;
        }
	private:
        vec3 albedo;
};

class metal : public material{
    public:
        __device__ metal(const vec3 &a, float f): albedo(a){
			if( f < 1.0f ){
				fuzz = f;
			}
			else{
				fuzz = 1;
			}		
		}
        __device__ virtual bool scatter(const ray &rIn, const hitRecord &rec, vec3 &attenuation, ray &scattered, curandState *localRandState) const{
            vec3 reflected = reflect(unitVector(rIn.direction()), rec.normal);
			scattered = ray(rec.p, unitVector(reflected + fuzz * randomInUnitSphere(localRandState)));
            attenuation = albedo;
            return (dot(scattered.direction(), rec.normal) > 0);
        }
    private:
        vec3 albedo;
		float fuzz;
};

#endif
