#ifndef HITABLELIST_H
#define HITABLELIST_H
 
#include "mkHitable.h"
 
//MK: __device__를 추가해서 GPU에서 호출가능 하도록 함수를 변경함
class hitableList: public hitable{
    public: 
        __device__ hitableList(){}

        __device__ hitableList(hitable **l, int n){
            list = l;
            listSize = n;
        }

        __device__ virtual bool hit(const ray &r, float tMin, float tMax, hitRecord &rec) const {
            hitRecord tempRec;
            bool hitAnything = false;
            double closestSoFar = tMax;
            for(int i = 0; i < listSize; i++){
                if(list[i]->hit(r, tMin, closestSoFar, tempRec)){
                    hitAnything = true;
                    closestSoFar = tempRec.t;
                    rec = tempRec;
                }
            }
            return hitAnything;
        }

    private:
        hitable **list;
        int listSize;
 
};
 
#endif
