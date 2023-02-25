// program to render the 3D matrix of a DLA simulation
#include <GL/glut.h>
#include <stdio.h>
#include <stdlib.h>


int*** matrix;
int size = 0;

int width = 2000, height = 1000;
int lastX = -1, lastY = -1;
float angleX = 0.0, angleY = 0.0;
float distance = 10.0;

int*** read_matrix_from_file(int* dim, char* filename) {
    FILE* fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("Failed to open file for reading.\n");
        return NULL;
    }
    fscanf(fp, "%d", dim);

    int ***matrix = (int ***)malloc((*dim) * sizeof(int **));
    for (int i = 0; i < (*dim); i++) {
        matrix[i] = (int **)malloc((*dim) * sizeof(int *));
        for (int j = 0; j < (*dim); j++)
            matrix[i][j] = (int *)calloc((*dim), sizeof(int));
    }
    
    int x, y, z;
    while (fscanf(fp, "%d %d %d", &x, &y, &z) != EOF) 
        matrix[x][y][z] = 1;
    
    fclose(fp);
    return matrix;
}


void free_matrix(int*** matrix, int dim) {
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) 
            free(matrix[i][j]);
		free(matrix[i]);
	}
	free(matrix);
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // set camera position
    glTranslatef(-(size-1)/2.0, -(size-1)/2.0, -(size-1)/2.0 - distance);
    glRotatef(angleY, 1.0, 0.0, 0.0);
    glRotatef(angleX, 0.0, 1.0, 0.0);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++) {
                glPushMatrix();
                glTranslatef(i-1.0, j-1.0, k-1.0);
                if(matrix[i][j][k]) {
                    // Draw solid cube
                    if(i == size / 2 && j == size / 2 && k == size / 2) {
                        // Draw borders of cube using a blue wireframe
                        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
                        glColor3f(0.0, 0.0, 1.0);
                        glLineWidth(2.0); // Set the line width to 2
                        glutWireCube(size);

                        // Reset polygon mode and color
                        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
                    }
                    glColor3f(1.0, 1.0, 1.0); // Set the color of the cube faces to white
                    glutSolidCube(0.99);

                    // Draw wireframe cube
                    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
                    glColor3f(0.5, 0.5, 0.5); // Set the color of the wireframe to gray
                    glLineWidth(2.0); // Set the line width to 2
                    glutWireCube(1.0);
                    
                    // Reset polygon mode and color
                    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
                    glColor3f(1.0, 1.0, 1.0);
                }
                
                glPopMatrix();
            }
        }
    }
    glutSwapBuffers();

    // draw borders of matrix using a red cube
    glLineWidth(3.0); // set line width to 3
    glColor3f(1.0, 0.0, 0.0); // set color to red
    glutWireCube(size); // draw wireframe cube with size of matrix

}

void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        lastX = x;
        lastY = y;
    }
    // Handle mouse wheel events
    if (button == 3) { // Mouse wheel up
        distance -= 0.5;
        glutPostRedisplay();
    }
    else if (button == 4) { // Mouse wheel down
        distance += 0.5;
        glutPostRedisplay();
    }
}

void motion(int x, int y) {
    if (lastX != -1 && lastY != -1) {
        angleX += (x - lastX) * 0.1;
        angleY += (y - lastY) * 0.1;
        lastX = x;
        lastY = y;
        glutPostRedisplay();
    }
}

void keyboard(unsigned char key, int x, int y) {
    if (key == 'f') { // zoom in
        distance -= 0.5;
        glutPostRedisplay();
    }
    else if (key == 's') { // zoom out
        distance += 0.5;
        glutPostRedisplay();
    }
}

int main(int argc, char** argv) {
    // load matrix from file
    matrix = read_matrix_from_file(&size, "matrix.txt");

    distance = size;

    // OpenGL setup
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(width, height);
    glutCreateWindow("3D Object from 3D Matrix");
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    // gluPerspective(45.0, 1.0, 0.1, (double) (size * 5));
    gluPerspective(45.0, (double) width / height, 0.1, (double) (size * 5));
    glutDisplayFunc(display);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(keyboard);
    glEnable(GL_DEPTH_TEST);
    glutMainLoop();

    free_matrix(matrix, size);

    return 0;
}

