#include <fstream>
#include "mkRay.h"
#include <time.h>
 
using namespace std;
 
bool hitSphere(const vec3 &center, float radius, const ray &r){
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = 2.0 * dot(oc, r.direction());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b*b - 4*a*c;
    return (discriminant >= 0);
}
 
vec3 color(const ray &r){
    vec3 ret = vec3(1, 0, 0);
    if(hitSphere(vec3(0, 0, -1), 0.5, r)){
        return ret;
    }
    vec3 unitDirection = unitVector(r.direction());
    float t = 0.5 * (unitDirection.y() + 1.0);
    ret = (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
    return ret;
}

int main(){
    int nx = 1200;
    int ny = 600;
    
    clock_t start, stop;
    start = clock();

    string fileName = "Ch4_cpu.ppm";
    ofstream writeFile(fileName.data());
    if(writeFile.is_open()){
        writeFile.flush();
        writeFile << "P3\n" << nx << " " << ny << "\n255\n";
        vec3 lowerLeftCorner(-2.0, -1.0, -1.0);
        vec3 horizontal(4.0, 0.0, 0.0);
        vec3 vertical(0.0, 2.0, 0.0);
        vec3 origin(0.0, 0.0, 0.0);
        for(int j = ny - 1; j >= 0; j--){
            for(int i = 0; i < nx; i++){
                float u = float(i) / float(nx);
                float v = float(j) / float(ny);
                ray r(origin, (lowerLeftCorner + u * horizontal + v * vertical));
                vec3 col = color(r);
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

