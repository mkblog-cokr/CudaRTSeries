#include <fstream>
#include "mkRay.h"
#include <time.h>
#include "mkHitablelist.h"
#include "float.h"
#include "mkSphere.h"
#include "mkCamera.h"

using namespace std;

vec3 randomInUnitSphere(){
    vec3 ret;
    do{
        ret = 2.0 * vec3(drand48(), drand48(), drand48()) - vec3(1, 1, 1);
    }while(ret.squared_length() >= 1.0);
    return ret;
}

vec3 color(const ray &r, hitable *world, int depth){
    hitRecord rec;
	//MK: Shadow Acne Problem 제거 코드
    if(depth < 50 && world->hit(r, 0.001, MAXFLOAT, rec)){
        vec3 target = rec.p + rec.normal + randomInUnitSphere();
        return 0.5 * color( ray(rec.p, unitVector(target-rec.p)), world, depth + 1 );
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
    int ns = 100;
	
	clock_t start, stop;
	start = clock();

    string fileName = "Ch7_cpu.ppm";
    ofstream writeFile(fileName.data());
    if(writeFile.is_open()){
        writeFile.flush();
        writeFile << "P3\n" << nx << " " << ny << "\n255\n";
        hitable *list[2];
        list[0] = new sphere(vec3(0, 0, -1), 0.5);
        list[1] = new sphere(vec3(0, -100.5, -1), 100);
        hitable *world = new hitableList(list, 2);
        camera cam;
        for(int j = ny - 1; j >= 0; j--){
            for(int i = 0; i < nx; i++){
                //MK: Sampling을 통하여 Pixel 값을 결정 (Anti-Aliasing)
                vec3 col(0.0, 0.0, 0.0);
                for(int s = 0; s < ns; s++){
                    float u = float(i + drand48()) / float(nx);
                    float v = float(j + drand48()) / float(ny);
                    ray r = cam.getRay(u, v);
                    col += color(r, world, 0);
                }
                col /= float(ns);
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

