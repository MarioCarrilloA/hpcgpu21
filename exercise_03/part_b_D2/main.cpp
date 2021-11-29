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

void simulate(uchar3* ptr, int tick, int width, int height, p *particles);


void init(p *particles) {
    int i;
    srand (time(NULL));
    for (i=0; i < MAX_PARTICLES; i++) {
        particles[i].angle = (float(rand())/float((RAND_MAX)) * 360.0); // 0 - 360
        particles[i].x = SUN_POS_X + SUN_RADIUS * cos(particles[i].angle);
        particles[i].y = SUN_POS_Y + SUN_RADIUS * sin(particles[i].angle);
        particles[i].mass = rand() % 10 + 1; // 1 - 10
        //particles[i].radius = rand() % 5 + 1; // 1 - 5
        //particles[i].v0 = (float(rand())/float((RAND_MAX)) * 100.0); // 0 - 100
        particles[i].vx0 = (float(rand())/float((RAND_MAX)) * 100.0); // 0 - 100
        particles[i].vx0 = particles[i].vx0 * cos(particles[i].angle);
        particles[i].vy0 = (float(rand())/float((RAND_MAX)) * 100.0); // 0 - 100
        particles[i].vy0 = particles[i].vy0 * sin(particles[i].angle);
        particles[i].default_y = particles[i].y;
        particles[i].default_x = particles[i].x;
        particles[i].default_vx0 = particles[i].vx0;
        particles[i].default_vy0 = particles[i].vy0;
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
	simulate(ptr, tick, dimx, dimy, particles);
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
