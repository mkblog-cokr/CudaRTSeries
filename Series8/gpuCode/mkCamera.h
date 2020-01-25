#ifndef MKCAMERA_H
#define MKCAMERA_H

#include "mkRay.h"

//MK: (코드 1) CPU Camera 코드를 복사후 GPU에서 실행하기 위해서 함수 이름 앞에 __device__를 추가함
class camera {
	public:
		__device__ camera() {
			lower_left_corner = vec3(-2.0, -1.0, -1.0);
			horizontal = vec3(4.0, 0.0, 0.0);
			vertical = vec3(0.0, 2.0, 0.0);
			origin = vec3(0.0, 0.0, 0.0);
		}

		__device__ ray get_ray(float u, float v) { 
			return ray(origin, lower_left_corner + u*horizontal + v*vertical - origin); 
		}
	
	private:
        vec3 origin;
        vec3 lower_left_corner;
        vec3 horizontal;
        vec3 vertical;
};

#endif
