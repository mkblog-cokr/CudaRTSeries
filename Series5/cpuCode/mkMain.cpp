#include <fstream>
#include "mkRay.h"
#include <time.h>
#include "mkHitablelist.h"
#include "float.h"
#include "mkSphere.h"
 
using namespace std;

//MK: 배경 및 Ray가 도달하는 위치의 색상을 결정함
vec3 color(const ray &r, hitable *world){
    hitRecord rec;
    if(world->hit(r, 0.0, MAXFLOAT, rec)){
        return 0.5 * vec3(rec.normal.x() + 1, rec.normal.y() + 1, rec.normal.z() + 1);
    }
    else{
        vec3 unitDirection = unitVector(r.direction());
        float t = 0.5 * (unitDirection.y() + 1.0);
        return (1.0 - t)*vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
    }
}

int main(){
    int nx = 1200;
    int ny = 600;
    
    clock_t start, stop;
    start = clock();

    string fileName = "Ch5_cpu.ppm";
    ofstream writeFile(fileName.data());
    if(writeFile.is_open()){
        writeFile.flush();
        writeFile << "P3\n" << nx << " " << ny << "\n255\n";
        vec3 lowerLeftCorner(-2.0, -1.0, -1.0);
        vec3 horizontal(4.0, 0.0, 0.0);
        vec3 vertical(0.0, 2.0, 0.0);
        vec3 origin(0.0, 0.0, 0.0);

		hitable *list[2];
		
		//MK: 2개의 구를 추가하는 부분
        list[0] = new sphere(vec3(0, 0, -1), 0.5);
        list[1] = new sphere(vec3(0, -100.5, -1), 100);
        //MK: 모든 Hitable Object를 hitableList에 추가함
        hitable *world = new hitableList(list, 2);
        for(int j = ny - 1; j >= 0; j--){
            for(int i = 0; i < nx; i++){
                float u = float(i) / float(nx);
                float v = float(j) / float(ny);
                ray r(origin, (lowerLeftCorner + u * horizontal + v * vertical));
                vec3 col = color(r, world);
                int ir = int(255.99 * col[0]);
                int ig = int(255.99 * col[1]);
                int ib = int(255.99 * col[2]);
                writeFile << ir << " " << ig << " " << ib << "\n";
            }
        }
        writeFile.close();
    }
    
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    cout << "MK: CPU Took " << timer_seconds << " Seconds.\n";

    return 0;
}

