// program to render the 3D matrix of a DLA simulation
#include <GL/glut.h>
#include <stdio.h>
#include <stdlib.h>

int size = 0;
int* points;
int lines = 0;

int width = 2000, height = 1000;
int lastX = -1, lastY = -1;
float angleX = 0.0, angleY = 0.0;
float distance = 10.0;

int count_lines(char* filename) {
    int count = 0;
    char ch;
    FILE* fp;

    fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("Error opening file!\n");
        return 0;
    }

    while ((ch = fgetc(fp)) != EOF)
        if (ch == '\n')
            count++;

    fclose(fp);
    return count - 1;
}

// function that reads the points from the file and stores them in an array
int* read_points_from_file(int* dim, char* filename) {
    lines = count_lines(filename);

    FILE* fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("Failed to open file for reading.\n");
        return NULL;
    }
    fscanf(fp, "%d", dim);

    points = (int*)malloc(lines * 3 * sizeof(int));
    int x, y, z;
    int i = 0;
    while (fscanf(fp, "%d %d %d", &x, &y, &z) != EOF) {
        points[i++] = x;
        points[i++] = y;
        points[i++] = z;
    }

    fclose(fp);

    return points;
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // set camera position
    glTranslatef(0.0, 0.0, -distance);
    glRotatef(angleY, 1.0, 0.0, 0.0);
    glRotatef(angleX, 0.0, 1.0, 0.0);
    glTranslatef(-(size - 1) / 2.0, -(size - 1) / 2.0, -(size - 1) / 2.0);
    // draw points
    for (int i = 0; i < lines; i++) {
        // Draw solid cube
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glPushMatrix();
        glTranslatef(points[i * 3], points[i * 3 + 1], points[i * 3 + 2]);
        glColor3f(1.0, 1.0, 1.0);
        glutSolidCube(0.99);
        // Draw wireframe cube
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glColor3f(0.5, 0.5, 0.5);  // Set the color of the wireframe to gray
        glLineWidth(2.0);          // Set the line width to 2
        glutWireCube(1.0);
        glPopMatrix();
    }

    // draw borders of matrix using a red cube
    // glLoadIdentity();
    glTranslatef((size - 1) / 2.0, (size - 1) / 2.0, (size - 1) / 2.0);
    glLineWidth(3.0);          // set line width to 3
    glColor3f(0.0, 0.0, 1.0);  // set color to blue
    glutWireCube(size);        // draw wireframe cube with size of matrix
    glutSwapBuffers();
}

void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        lastX = x;
        lastY = y;
    }
    // Handle mouse wheel events
    if (button == 3) {  // Mouse wheel up
        distance -= 2.0;
        glutPostRedisplay();
    } else if (button == 4) {  // Mouse wheel down
        distance += 2.0;
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
    if (key == 'i') {  // zoom in
        distance -= 2.0;
        glutPostRedisplay();
    } else if (key == 'o') {  // zoom out
        distance += 2.0;
        glutPostRedisplay();
    }
}

int main(int argc, char** argv) {
    // load matrix from file
    points = read_points_from_file(&size, "matrix.txt");

    distance = size * 1.3;

    // OpenGL setup
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(width, height);
    glutCreateWindow("3D Object from 3D Matrix");
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (double)width / height, 0.1, (double)(size * 5));
    glutDisplayFunc(display);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(keyboard);
    glEnable(GL_DEPTH_TEST);
    glutMainLoop();

    free(points);

    return 0;
}