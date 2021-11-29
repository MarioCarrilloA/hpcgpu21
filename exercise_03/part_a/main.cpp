#ifdef _WIN64
#define GLUT_NO_LIB_PRAGMA
#pragma comment (lib, "opengl32.lib")  /* link with Microsoft OpenGL lib */
#pragma comment (lib, "glut64.lib")    /* link with Win64 GLUT lib */
#pragma comment (lib, "winmm.lib")    /* link with Win64 GLUT lib */
//_WIN64
#include <Windows.h>
#endif


#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/glext.h>
#include <stdio.h>
/*
#include "Utils.h"
*/
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include "utils.h"


GLuint  vbo;
cudaGraphicsResource *resource;
size_t  size;
uchar3* ptr;
int dimx,dimy;
int tick=0,A_ex=0;
p particles[MAX_PARTICLES];

void simulate_p1(uchar3* ptr, int tick, int width, int height, p *particles);

/**
 * For the moment, a particle can only take integer values
 * between range 0 - 10 for mass. This function assigns a color
 * according to that value.
 */
void set_colors(p *particles) {
    int mass = particles->mass;
    int *red = &particles->red;
    int *green = &particles->green;
    int *blue = &particles->blue;

    switch(mass) {
        case 1:
            *red=255, *green=241, *blue=0;
        break;
       case 2:
            *red=255, *green=140, *blue=0;
        break;
        case 3:
            *red=232, *green=17, *blue=35;
        break;
        case 4:
            *red=236, *green=0, *blue=140;
        break;
        case 5:
            *red=104, *green=33, *blue=122;
        break;
        case 6:
            *red=0, *green=24, *blue=143;
        break;
        case 7:
            *red=0, *green=188, *blue=242;
        break;
        case 8:
            *red=0, *green=178, *blue=148;
        break;
        case 9:
            *red=0, *green=158, *blue=73;
        break;
        case 10:
            *red=186, *green=216, *blue=10;
        break;
    }
}

/**
 * This function assing random values for MAX_PARTICLES
 * number of particles.
 */
void init(p *particles) {
    int i;
    srand (time(NULL));
    for (i=0; i < MAX_PARTICLES; i++) {
        particles[i].x = rand() % 800 + 1;  // 0 - 800
        particles[i].y = rand() % 20 + 781; // 750 - 800
        particles[i].mass = rand() % 10 + 1;
        particles[i].radius = rand() % 10 + 1; // 1 - 10
        particles[i].v0 = (float(rand())/float((RAND_MAX)) * 200.0); // 0 - 200
        particles[i].default_y = particles[i].y;
        particles[i].default_v0 = particles[i].v0;
        set_colors(&particles[i]);
    }
}


void initialize(int *width, int *height) {
	// initialize the dimension of the picture here
	*width=800;
	*height=800;
}

void key( unsigned char key, int x, int y )
{
    switch (key) {
        case 27: // On Escape Key
		default:
            // Clean up OpenGL and CUDA ressources
            checkCudaErrors(cudaGraphicsUnregisterResource(resource));
	    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
            glDeleteBuffersARB(1, &vbo);
            exit(0);
    }
}

void display()
{
	// Map the device memory to CUDA ressource pointer
	checkCudaErrors(cudaGraphicsMapResources(1, &resource, NULL));
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&ptr, &size, resource));
	printf("%d ist size bei %d \n",size,dimx*dimy);

	// Execute the CUDA kernel
	simulate_p1(ptr, tick, dimx, dimy, particles);
	tick++;

	// Unmap the ressource for visualization
	checkCudaErrors(cudaGraphicsUnmapResources(1, &resource, NULL));

	// Draw the VBO as bitmap
    	glDrawPixels(dimx, dimy, GL_RGB, GL_UNSIGNED_BYTE, 0);

	// Swap OpenGL buffers
    	glutSwapBuffers();
}

void generate(int w, int h)
{
	// Create standard OpenGL 1.5 Vertex Buffer Object
	glDeleteBuffersARB(1, &vbo);
	glGenBuffersARB(1, &vbo);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, vbo);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, w * h * 3, NULL, GL_DYNAMIC_DRAW_ARB);

	// Register buffer for CUDA access
	 checkCudaErrors(cudaGraphicsGLRegisterBuffer(&resource, vbo,cudaGraphicsMapFlagsNone));
}

void reshape(int w, int h)
{
	// Set global dimensions
	dimx = w; dimy = h;

	// Generate GL buffer and register VBO
	generate(w, h);

	// Reset the projection for resize.
	glViewport(0,0,(GLsizei)w,(GLsizei)h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, (GLfloat)w / (GLfloat)h, 0.1, 2000);
	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char **argv)
{
	// Coose the CUDA / OpenGL device
	 checkCudaErrors(cudaSetDevice(0));

	// Initialize OpenGL over GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	// Initialize by user
	initialize(&dimx, &dimy);
    glutInitWindowSize(dimx, dimy);
    glutCreateWindow("CUDA Exercise");
	//printf("ein uchar ist %d und uchar4 sind %d\n",sizeof(unsigned char),sizeof(uchar4));
	//printf("Im Vergleich dazu ein int %d\n",sizeof(int));
	//getchar();
	// Extensions initialization
	glewInit();

    // Initial random values for particles
    init(particles);

	// Generate GL buffer and register VBO
    generate(dimx, dimy);


    // Register callbacks and start the GLUT render loop
    glutKeyboardFunc(key);
    glutDisplayFunc(display);
	glutIdleFunc(glutPostRedisplay);
	glutReshapeFunc(reshape);
    glutMainLoop();
}
