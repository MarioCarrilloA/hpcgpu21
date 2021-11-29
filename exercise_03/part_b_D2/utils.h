#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define MAX_PARTICLES 5000

// Sun
#define SUN_RADIUS    80
#define SUN_POS_X     400
#define SUN_POS_Y     400
#define ORBIT_POS_X   400
#define ORBIT_POS_Y   400
#define ORBIT_RADIUS   300

struct p {
    int x;
    int y;
    int mass;
    int radius;
    float degree;
    float ax;
    float ay;
    float vx0;
    float vy0;
    int default_x;
    int default_y;
    float default_vx0;
    float default_vy0;
};

