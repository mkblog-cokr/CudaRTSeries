#ifndef MKMATERIAL_H
#define MKMATERIAL_H

#include "mkVec3.h"

class material{
    public:
        virtual bool scatter(const ray &rIn, const hitRecord &rec, vec3 &attenuation, ray &scattered) const = 0;
};

class lambertian : public material{
    public:
        lambertian(const vec3 &a): albedo(a) {}
        virtual bool scatter(const ray &rIn, const hitRecord &rec, vec3 &attenuation, ray &scattered) const {
            vec3 target = rec.p + rec.normal + randomInUnitSphere();
            scattered = ray(rec.p, unitVector(target-rec.p));
            attenuation = albedo;
            return true;
        }
    private:
        vec3 albedo;
};

class metal : public material{
    public:
        metal(const vec3 &a, float f): albedo(a){
			if( f < 1.0f ){
				fuzz = f;
			}
			else{
				fuzz = 1;
			}		
		}
        virtual bool scatter(const ray &rIn, const hitRecord &rec, vec3 &attenuation, ray &scattered) const{
            vec3 reflected = reflect(unitVector(rIn.direction()), rec.normal);
            scattered = ray(rec.p, unitVector(reflected+fuzz*randomInUnitSphere()));
            attenuation = albedo;
            return (dot(scattered.direction(), rec.normal) > 0);
        }
    private:
        vec3 albedo;
		float fuzz;
};

#endif
