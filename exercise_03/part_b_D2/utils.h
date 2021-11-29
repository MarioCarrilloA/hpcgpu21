#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define MAX_PARTICLES 5000

// Sun
#define SUN_RADIUS    110
#define SUN_POS_X     200
#define SUN_POS_Y     600

// Earth
//#define EARTH_MASS    59720000
#define EARTH_MASS    (5972 * pow(10.0, 24.0))
#define EARTH_RADIUS  40
#define EARTH_POS_X   600
#define EARTH_POS_Y   200


struct p {
    int x;
    int y;
    int mass;
    //int radius;
    float angle;
    float ax;
    float ay;
    float vx0;
    float vy0;
    int default_x;
    int default_y;
    float default_vx0;
    float default_vy0;
};

