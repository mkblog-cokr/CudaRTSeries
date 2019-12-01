#ifndef HITABLELIST_H
#define HITABLELIST_H
 
#include "mkHitable.h"
 
//MK: 모든 Hitable Object를 리스트로 가지고 있음
//MK: Ray와 모든 Ojbect의 Hit(Intersection)여부를 판단함
class hitableList: public hitable{
    public: 
        hitableList(){}
        hitableList(hitable **l, int n){
            list = l;
            listSize = n;
        }

        virtual bool hit(const ray &r, float tMin, float tMax, hitRecord &rec) const {
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
