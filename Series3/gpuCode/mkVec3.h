#ifndef MKVEC3_H
#define MKVEC3_H
 
#include <math.h>
#include <stdlib.h>
#include <iostream>

//MK: (코드 1-1) __host__, __device__를 추가해서 CPU/GPU 모두 호출가능하도록 수정함
//MK: 그 이외 모든 부분을 기존 CPU코드와 동일함
class vec3{
    public:
      __host__ __device__ vec3(){}
      __host__ __device__ vec3(float e0, float e1, float e2){
        element[0] = e0;
        element[1] = e1;
        element[2] = e2;
      }
 
      __host__ __device__ inline float x() const{ return element[0];}
      __host__ __device__ inline float y() const{ return element[1];}
      __host__ __device__ inline float z() const{ return element[2];}
 
      __host__ __device__ inline float r() const{ return element[0];}
      __host__ __device__ inline float g() const{ return element[1];}
      __host__ __device__ inline float b() const{ return element[2];}
 
      __host__ __device__ inline const vec3& operator+() const{ return *this;}
      __host__ __device__ inline vec3 operator-() const {return vec3(-element[0], -element[1], -element[2]);}
      __host__ __device__ inline float operator[] (int i) const {return element[i];}
      __host__ __device__ inline float &operator[] (int i) {return element[i];}
 
      __host__ __device__ inline vec3& operator+=(const vec3 &v){
          element[0] += v.element[0];
          element[1] += v.element[1];
          element[2] += v.element[2];
          return *this;
      }

      __host__ __device__ inline vec3& operator-=(const vec3 &v){
          element[0] -= v.element[0];
          element[1] -= v.element[1];
          element[2] -= v.element[2];
          return *this;
      }

      __host__ __device__ inline vec3& operator*=(const vec3 &v){
          element[0] *= v.element[0];
          element[1] *= v.element[1];
          element[2] *= v.element[2];
          return *this;
      }

      __host__ __device__ inline vec3& operator/=(const vec3 &v){
          element[0] /= v.element[0];
          element[1] /= v.element[1];
          element[2] /= v.element[2];
          return *this;
      }

      __host__ __device__ inline vec3& operator*=(const float t){
          element[0] *= t;
          element[1] *= t;
          element[2] *= t;
          return *this;
      }

      __host__ __device__ inline vec3& operator/=(const float t){
          float k = 1.0/t;
          element[0] *= k;
          element[1] *= k;
          element[2] *= k;
          return *this;
      }
 
      __host__ __device__ inline float length() const{
          return sqrt(element[0] * element[0] + element[1] * element[1] + element[2] * element[2]);
      }

      __host__ __device__ inline float squared_length() const{
          return (element[0] * element[0] + element[1] * element[1] + element[2] * element[2]);
      }

	  __host__ __device__ inline void make_unit_vector(){
          float k = 1.0 / (sqrt(element[0] * element[0] + element[1] * element[1] + element[2] * element[2]));
          element[0] *= k;
          element[1] *= k;
          element[2] *= k;
      };
 
      float element[3];
};
 
inline std::istream& operator>>(std::istream &is, vec3 &t){
    is >> t.element[0] >> t.element[1] >> t.element[2];
    return is;
}
 
inline std::ostream& operator<<(std::ostream &os, const vec3 &t){
    os << t.element[0] << t.element[1] << t.element[2];
    return os;
}
 
__host__ __device__ inline vec3 operator+(const vec3 &v1, const vec3 &v2){
    return vec3(v1.element[0] + v2.element[0], v1.element[1] + v2.element[1], v1.element[2] + v2.element[2]);
}
 
__host__ __device__ inline vec3 operator-(const vec3 &v1, const vec3 &v2){
    return vec3(v1.element[0] - v2.element[0], v1.element[1] - v2.element[1], v1.element[2] - v2.element[2]);
}
 
__host__ __device__ inline vec3 operator*(const vec3 &v1, const vec3 &v2){
    return vec3(v1.element[0] * v2.element[0], v1.element[1] * v2.element[1], v1.element[2] * v2.element[2]);
}
 
__host__ __device__ inline vec3 operator/(const vec3 &v1, const vec3 &v2){
    return vec3(v1.element[0] / v2.element[0], v1.element[1] / v2.element[1], v1.element[2] / v2.element[2]);
}

__host__ __device__ inline vec3 operator*(const float t, const vec3 &v){
    return vec3(t * v.element[0], t * v.element[1], t * v.element[2]);
}
 
__host__ __device__ inline vec3 operator/(const vec3 &v, const float t){
    return vec3(v.element[0]/t, v.element[1]/t, v.element[2]/t);
}
 
__host__ __device__ inline vec3 operator*(const vec3 &v, const float t){
    return vec3(v.element[0] * t, v.element[1] * t, v.element[2] * t);
}
 
__host__ __device__ inline float dot(const vec3 &v1, const vec3 &v2){
    return (v1.element[0] * v2.element[0] + v1.element[1] * v2.element[1] + v1.element[2] * v2.element[2]);
}
 
__host__ __device__ inline vec3 cross(const vec3 &v1, const vec3 &v2){
    return vec3(
                (v1.element[1] * v2.element[2] - v1.element[2] * v2.element[1]),
                -(v1.element[0] * v2.element[2] - v1.element[2] * v2.element[0]),
                (v1.element[0] * v2.element[1] - v1.element[1] * v2.element[0])
            );
}
 
__host__ __device__ inline vec3 unitVector(vec3 v){
    return (v/v.length());
}
 
#endif
