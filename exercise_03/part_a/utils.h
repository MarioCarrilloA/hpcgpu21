#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define MAX_PARTICLES 5000
#define GRAVITY 9.8

struct p {
    int x;
    int y;
    int mass;
    int radius;
    int red;
    int green;
    int blue;
    float v0;
    int default_y;
    int default_v0;
};

