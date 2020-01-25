#include <fstream>
#include "mkRay.h"
#include <time.h>
#include "mkHitablelist.h"
#include "float.h"
#include "mkSphere.h"
#include "mkCamera.h"
#include "mkMaterial.h"

using namespace std;

vec3 color(const ray &r, hitable *world, int depth){
    hitRecord rec;
    if(world->hit(r, 0.001, MAXFLOAT, rec)){
        ray scattered;
        vec3 attenuation;
        if(depth < 50 && rec.matPtr->scatter(r, rec, attenuation, scattered)){
            return attenuation * color(scattered, world, depth + 1);
        }
        else{
            return vec3(0, 0, 0);
        }
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

    string fileName = "Ch8_cpu.ppm";
    ofstream writeFile(fileName.data());
    if(writeFile.is_open()){
        writeFile.flush();
        writeFile << "P3\n" << nx << " " << ny << "\n255\n";
        hitable *list[4];
		list[0] = new sphere(vec3(0, 0, -1), 0.5, new lambertian(vec3(0.8, 0.3, 0.3)));
        list[1] = new sphere(vec3(0, -100.5, -1), 100, new lambertian(vec3(0.8, 0.8, 0.0)));
        list[2] = new sphere(vec3(1, 0, -1), 0.5, new metal(vec3(0.8, 0.6, 0.2), 0.3f));
        list[3] = new sphere(vec3(-1, 0, -1), 0.5, new metal(vec3(0.8, 0.8, 0.8), 1.0f));
        hitable *world = new hitableList(list, 4);
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

