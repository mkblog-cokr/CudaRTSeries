#include <iostream>
#include <fstream>
#include <time.h>
#include "mkVec3.h"
 
using namespace std;
 
int main(){
	int nx = 1200;
	int ny = 600;
	
	string fileName = "Ch3_cpu.ppm";
	ofstream writeFile(fileName.data());
	
	clock_t start, stop;
	start = clock();
	
	if(writeFile.is_open()){
		writeFile.flush();
		writeFile << "P3\n" << nx << " " << ny << "\n255\n";
		for(int j = ny - 1; j >= 0; j--){
			for(int i = 0; i < nx; i++){
				vec3 col(float(i)/float(nx), float(j)/float(ny), 0.2);
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
