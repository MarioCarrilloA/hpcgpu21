#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define MAX_PARTICLES 10000
#define GRAVITY 9.8
#define PI 3.141592

struct p {
    int x;
    int y;
    int mass;
    int radius;
    int red;
    int green;
    int blue;
    float v0;
    int default_x;
    int default_y;
    int default_v0;
};

struct Earth {
    int xc;
    int yc;
    int radius;
    int distanceToSun;
};