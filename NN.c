#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Struct definitions
typedef struct {
    double bias;
    double * weights;
} Node;

typedef struct Layer_s {
    int node_count;
    Node ** nodes;
    struct Layer_s * next_layer;
    struct Layer_s * prev_layer;
} Layer;

typedef struct {
    int layers;
    double learning_rate;
    Layer * input_layer;
} NN;

typedef struct {
    int rows;
    int cols;
    double ** array;
} Matrix;

// Matrix operations
Matrix * dot_product(Matrix * m1, Matrix * m2);
Matrix * alloc_matrix(int rows, int cols);
void random_populate(Matrix * matrix);
Matrix * rand_matrix(int rows, int cols);
void display_matrix(Matrix * matrix);
double rand_in_range(double min, double max);
void zero_init(Matrix * matrix);
Matrix * transpose(Matrix * matrix);
Matrix * copy_matrix(Matrix * matrix);
Matrix * array_to_matrix(double ** array, int rows, int cols);
void print_shape(Matrix * matrix);

// Element-wise matrix operations
void element_wise_addition(Matrix * m1, Matrix * m2);
void element_wise_multiplication(Matrix * m1, Matrix * m2);
void element_wise_subtraction(Matrix * m1, Matrix * m2);
void element_wise_division(Matrix * m1, Matrix * m2);
int element_wiseable(Matrix * m1, Matrix * m2);

// Layer Operations
Layer * init_layer(int node_count);
Node * init_node();
void connect_layers(Layer * layer1, Layer * layer2);
double * init_weights(int nodes);
Matrix * convert_to_matrix(Layer * layer);
Matrix * nodes_to_matrix(Layer * layer);
Matrix * layer_to_biases(Layer * layer);

// Neural Network Functions
NN * init_NN(double learning_rate);
void add_layer(NN * neural_network, int nodes);
Matrix * feed_forward(Matrix * input, NN * nn);
Matrix * feed_forward_internal(Matrix * input, Layer * current_layer);
Matrix * back_propagate(Matrix * output_error, double * learning_rate);
Matrix * rand_input(NN * nn);
void plus_biases(Matrix * matrix, Layer * layer);


// Display functions
void display_node(Node * node);
void display_layer(Layer * layer);
void display_NN(NN * nn);
void display_weights(Layer * layer);
void display_NN_with_weights(NN * nn);

// Free Functions
void free_NN(NN * nn);
void free_NN_layer(Layer * curr_layer);
void free_matrix(Matrix * matrix);

// Activation functions
void relu(Matrix * matrix);
void relu_prime(Matrix * matrix);

// Cost functions
double mean_squared_error(Matrix * y_pred, Matrix * y_true);

int main (void)
{
    srand(time(NULL));
    return 0;
}

Matrix * rand_matrix(int rows, int cols) {
    Matrix * matrix = alloc_matrix(rows, cols);
    random_populate(matrix);
    return matrix;
}

// Determines if the given matrices can be accessed element-wise
int element_wiseable(Matrix * m1, Matrix * m2) {
    return (m1->rows == m2->rows && m1->cols == m2->cols);
}

// Element-wise addition between matrices
void element_wise_addition(Matrix * m1, Matrix * m2) {

    if (!element_wiseable(m1, m2)) {
        printf("Mismatching dimensions for element-wise addition\n");
        exit(1);
    }
    int rows = m1->rows, cols = m1->cols;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            m1->array[i][j] += m2->array[i][j];
}

// Element-wise multiplication between matrices
void element_wise_multiplication(Matrix * m1, Matrix * m2) {

    if (!element_wiseable(m1, m2)) {
        printf("Mismatching dimensions for element-wise addition\n");
        exit(1);
    }
    int rows = m1->rows, cols = m2->cols;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            m1->array[i][j] *= m2->array[i][j];
}

// Element-wise division between matrices (order matters)
void element_wise_division(Matrix * m1, Matrix * m2) {

    if (!element_wiseable(m1, m2)) {
        printf("Mismatching dimensions for element-wise addition\n");
        exit(1);
    }
    int rows = m1->rows, cols = m2->cols;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            m1->array[i][j] /= m2->array[i][j];
}

// Element-wise subtraction between matrices (order matters)
void element_wise_subtraction(Matrix * m1, Matrix * m2) {
    if (!element_wiseable(m1, m2)) {
        printf("Mismatching dimensions for element-wise addition\n");
        exit(1);
    }
    int rows = m1->rows, cols = m2->cols;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            m1->array[i][j] -= m2->array[i][j];
}
/*
Matrix * back_propagate(NN * nn, Matrix * y_true, Matrix * y_pred) {
    Layer * current_layer = nn->input_layer;
    while (current_layer->next_layer != NULL)
        current_layer = current_layer->next_layer;

    Matrix * output_error
    while (current_layer->prev_layer != NULL) {

    }
    Matrix * output_error = mean_squared_error(y_pred, y_true);



}
*/

// Print the shape of the given matrix
void print_shape(Matrix * matrix) {
    printf("(%d, %d)\n", matrix->rows, matrix->cols);
}


// Random input based on the architecture of the Neural Network
Matrix * rand_input(NN * nn) {
    double num;
    Matrix * input = alloc_matrix(1, nn->input_layer->node_count);
    int rows = input->rows, cols = input->cols;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            input->array[i][j] = rand_in_range(-3.0, 3.0);
    return input;
}

// User friendly abstraction over feed_forward(input, nn->input_layer)
Matrix * feed_forward(Matrix * input, NN * nn) {
    return feed_forward_internal(input, nn->input_layer);
}

// The classic feed_forward algorithm.
// Implemented with recursion
Matrix * feed_forward_internal(Matrix * input, Layer * current_layer) {
    if (current_layer->next_layer == NULL)
        return input;

    Matrix * weights = convert_to_matrix(current_layer);
    Matrix * output = dot_product(input, weights);
    plus_biases(output, current_layer->next_layer);
    relu(output);
    free(weights);

    if (current_layer->prev_layer != NULL)
        free_matrix(input);

    return feed_forward_internal(output, current_layer->next_layer);
}

// Element-wise addition of matrices
void plus_biases(Matrix * matrix, Layer * layer) {

    if (matrix->cols == layer->node_count || matrix->rows == layer->node_count) {
        int cols = matrix->cols, rows = matrix->rows;
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                matrix->array[i][j] += layer->nodes[j]->bias;
    } else {
        printf("Cannot add biases of mismatching lengths!\n");
        exit(1);
    }

}

// Converts given double array to matrix
Matrix * array_to_matrix(double ** array, int rows, int cols) {
    Matrix * matrix = alloc_matrix(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < rows; j++)
            matrix->array[i][j] = array[i][j];
    return matrix;
}



// Converts the nodes in a layer to a matrix
Matrix * nodes_to_matrix(Layer * layer) {
    Matrix * matrix = alloc_matrix(layer->node_count, 1);
    int rows = matrix->rows, cols = matrix->cols;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            matrix->array[i][j] = layer->nodes[i]->bias;
    return matrix;
}
// (y_pred - y_true) ^ 2
double mean_squared_error(Matrix * y_pred, Matrix * y_true) {
    double mse;
    int rows = y_pred->rows, cols = y_pred->cols;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            mse += pow(y_true->array[i][j] - y_pred->array[i][j], 2);
    return mse;
}

//The Rectified Linear Unit function
void relu(Matrix * matrix) {
    int rows = matrix->rows, cols = matrix->cols;
    double num;
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            num = matrix->array[i][j];
            if (num <= 0)
                matrix->array[i][j] = 0;
        }
}

// Makes a full copy of the given matrix
Matrix * copy_matrix(Matrix * matrix) {
    Matrix * copy = alloc_matrix(matrix->rows, matrix->cols);
    int rows = matrix->rows, cols = matrix->cols;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            copy->array[i][j] = matrix->array[i][j];
    return copy;
}


// Makes an implicit matrix from a layer of dimensions
// (curr_layer->node_count X next_layer->node_count)
Matrix * convert_to_matrix(Layer * layer) {
    Matrix * matrix = alloc_matrix(layer->node_count, layer->next_layer->node_count);
    for (int i = 0; i < layer->node_count; i++)
        matrix->array[i] = layer->nodes[i]->weights;
    return matrix;
}

// User friendly abstraction over free_NN_layer
void free_NN(NN * nn) {
    free_NN_layer(nn->input_layer);
    free(nn->input_layer);
    free(nn);
}

// Display the weights of a layer
void display_weights(Layer * layer) {
    for (int i = 0; i < layer->node_count; i++) {
        printf("Node %d:\n", i);
        printf("[ ");
        for (int j = 0; j < layer->next_layer->node_count; j++)
            printf("%.3lf ", layer->nodes[i]->weights[j]);
        printf(" ]\n");
    }
}

// Recursive freeing function
void free_NN_layer(Layer * curr_layer) {
    if (curr_layer->next_layer != NULL)
        free_NN_layer(curr_layer->next_layer);

    for (int i = 0; i < curr_layer->node_count; i++) {
        free(curr_layer->nodes[i]->weights);
        curr_layer->nodes[i]->weights = NULL;
        free(curr_layer->nodes[i]);
        curr_layer->nodes[i] = NULL;
    }

    free(curr_layer->nodes);
    curr_layer->nodes = NULL;
    free(curr_layer);
    curr_layer = NULL;
    return;
}

// Display the nodes and their biases in a neural network
void display_NN(NN * nn) {
    if (nn == NULL) {
        printf("[NULL]\n");
        return;
    }
    Layer * temp = nn->input_layer;
    while (temp != NULL) {
        display_layer(temp);
        temp = temp->next_layer;
    }
    printf("\n\n");
}

// Display the nodes with their biases as well as connections
void display_NN_with_weights(NN * nn) {
    if (nn == NULL) {
        printf("[NULL]\n");
        return;
    }
    Layer * temp = nn->input_layer;
    for (int i = 0; i < nn->layers - 1; i++) {
        display_layer(temp);
        display_weights(temp);
        temp = temp->next_layer;
    }
    // The last layer doesn't have weights connected
    // to any further layers, but has itself a bias to
    // display
    display_layer(temp);
    printf("\n\n");
}

// Initialize a neural network struct
NN * init_NN(double learning_rate) {
    NN * nn = (NN *) malloc(sizeof(NN) * 1);
    nn->input_layer = NULL;
    nn->layers = 0;
    nn->learning_rate = learning_rate;
    return nn;
}

// Connect the previous last layer with a freshly initialized one
void add_layer(NN * nn, int nodes) {
    if (nn->input_layer == NULL) {
        nn->input_layer = init_layer(nodes);
        nn->layers += 1;
        return;
    } else {
        Layer * temp = nn->input_layer;
        while (temp->next_layer != NULL)
            temp = temp->next_layer;
        connect_layers(temp, init_layer(nodes));
        nn->layers += 1;
        return;
    }
}

// Connect the two layers with random weights.
// The randomization of weights is implicit because connecting
// layers only occurs during the creation of the network
void connect_layers(Layer * layer1, Layer * layer2) {
    layer1->next_layer = layer2;
    layer2->prev_layer = layer1;
    for (int i = 0; i < layer1->node_count; i++)
        layer1->nodes[i]->weights = init_weights(layer2->node_count);
}

// Initializes an array of random values
double * init_weights(int nodes) {
    double * weights = (double *) malloc(sizeof(double) * nodes);
    for (int i = 0; i < nodes; i++)
        weights[i] = rand_in_range(-0.1, 0.1);
    return weights;
}

// Simply display the bias of a node
void display_node(Node * node) {
    printf("[B:%.3lf]\n", node->bias);
}

// Display a layer and whether it has layers in front or behind it
void display_layer(Layer * layer) {
    if (layer->prev_layer != NULL)
        printf("Prev: Y\n");
    else
        printf("Prev: N\n");

    if (layer->next_layer != NULL)
        printf("Next: Y\n");
    else
        printf("Next: N\n");

    for (int i = 0; i < layer->node_count; i++)
        display_node(layer->nodes[i]);

}

// Initializes and allocates a Node * struct
Node * init_node() {
    Node * new_node = (Node *) malloc(sizeof(Node) * 1);
    new_node->bias = rand_in_range(-1.0, 1.0);
    new_node->weights = NULL;
    return new_node;
}

// Initializes and allocates a Layer * struct given the number of nodes
Layer * init_layer(int node_count) {
    Layer * new_layer = (Layer *) malloc(sizeof(Layer) * 1);
    new_layer->next_layer = NULL;
    new_layer->prev_layer = NULL;
    new_layer->node_count = node_count;
    new_layer->nodes = (Node **) malloc(sizeof(Node *) * node_count);
    for (int i = 0; i < node_count; i++)
        new_layer->nodes[i] = init_node();
    return new_layer;
}


/// Matrix operations
Matrix * transpose(Matrix * matrix) {
    Matrix * transposed_matrix = alloc_matrix(matrix->cols, matrix->rows);
    zero_init(transposed_matrix);
    int rows = transposed_matrix->rows, cols = transposed_matrix->cols;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            transposed_matrix->array[i][j] = matrix->array[j][i];

    free_matrix(matrix);
    return transposed_matrix;
}

// Initializes the given matrix with 0s
void zero_init(Matrix * matrix) {
    for (int r = 0; r < matrix->rows; r++)
        for (int c = 0; c < matrix->cols; c++)
            matrix->array[r][c] = 0;
}

// Displays the matrix given the dimensions
void display_matrix(Matrix * matrix) {
   double num = 0.0;
   print_shape(matrix);
   for (int i = 0; i < matrix->rows; ++i) {
      for (int j = 0; j < matrix->cols; ++j) {
         num = matrix->array[i][j];
         if (num < 0)
            printf("[%.2lf ] ", num);
         else
            printf("[ %.2lf ] ", num);
         if (j == matrix->cols - 1)
            printf("\n");
      }
   }
   printf("\n\n");
}

// Populates a matrix with random values (-1, 1)
void random_populate(Matrix * matrix) {
    int rows = matrix->rows, cols = matrix->cols;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            matrix->array[i][j] = rand_in_range(-1.0, 1.0);
}

// Generates a random double in the given range
double rand_in_range(double min, double max) {
    double random = ((double) rand()) / RAND_MAX;
    double range = (max - min) * random;
    double number = min + range;
    return number;
}

// Allocates a 2D matrix of doubles with specified dimensions
Matrix * alloc_matrix(int rows, int cols) {
    Matrix * matrix = (Matrix *) malloc(sizeof(Matrix) * 1);
    double ** array = (double **) malloc(sizeof(double *) * rows);
    for (int i = 0; i < rows; i++)
        array[i] = (double *) malloc(sizeof(double) * cols);


    matrix->array = array;
    matrix->rows = rows;
    matrix->cols = cols;
    zero_init(matrix);
    return matrix;
}

// Frees the given array with the given dimensions
void free_matrix(Matrix * matrix) {
    for (int i = 0; i < matrix->rows; i++)
        free(matrix->array[i]);
    free(matrix->array);
    free(matrix);
}

// Takes arrays with specified dimensions and returns the dot product
Matrix * dot_product(Matrix * m1, Matrix * m2)
{
    // Check for mismatching dimensions
    if (m1->cols != m2->rows) {
        printf("Mismatching matrices dimensions!\n");
        exit(1);
    }

    Matrix * result_matrix = alloc_matrix(m1->rows, m2->cols);
    zero_init(result_matrix);
    for (int i = 0; i < m1->rows; ++i)
        for (int j = 0; j < m2->cols; ++j)
            for (int k = 0; k < m1->cols; ++k)
                result_matrix->array[i][j] += m1->array[i][k] * m2->array[k][j];
    return result_matrix;
}
