#include <iostream>
#include <fstream>
#include <time.h>
 
using namespace std;
 
int main(){
	int nx = 1200;
	int ny = 600;
	
	string fileName = "Ch1_cpu.ppm";
	ofstream writeFile(fileName.data());
	
	clock_t start, stop;
	start = clock();
	
	if(writeFile.is_open()){
		writeFile.flush();
		writeFile << "P3\n" << nx << " " << ny << "\n255\n";
		for(int j = ny - 1; j >= 0; j--){
			for(int i = 0; i < nx; i++){
				float r = float(i) / float(nx);
				float g = float(j) / float (ny);
				float b = 0.2;
				int ir = int(255.99 * r);
				int ig = int(255.99 * g);
				int ib = int(255.99 * b);
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
