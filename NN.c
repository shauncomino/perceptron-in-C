#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

///                    ///
/// Struct definitions ///
///                    ///

typedef struct {
    int rows;
    int cols;
    double ** array;
} Matrix;

typedef struct {
    double bias;
    double * weights;
} Node;

typedef struct Layer_s {
    int node_count;
    Node ** nodes;
    Matrix * input;
    Matrix * output;
    struct Layer_s * next_layer;
    struct Layer_s * prev_layer;
} Layer;

typedef struct {
    int layers;
    double learning_rate;
    Layer * input_layer;
} NN;

typedef struct {
    Matrix ** x;
    Matrix ** y;
    int num_examples;
} Dataset;

typedef struct {
    Matrix * x;
    Matrix * y;
} Data;

// Matrix operations
Matrix * dot_product(Matrix * m1, Matrix * m2);
Matrix * alloc_matrix(int rows, int cols);
void random_populate(Matrix * matrix);
Matrix * rand_matrix(int rows, int cols);
void display_matrix(Matrix * matrix);
double rand_in_range(double min, double max);
void zero_init(Matrix * matrix);
Matrix * transpose(Matrix * matrix);
Matrix * transpose_destroy(Matrix * matrix);
Matrix * copy_matrix(Matrix * matrix);
void print_shape(Matrix * matrix);
void multiply_by (Matrix * matrix, double value);

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

// Neural Network Operations
NN * init_NN(double learning_rate);
void add_layer(NN * neural_network, int nodes);
Matrix * rand_input(NN * nn);
void plus_biases(Matrix * matrix, Layer * layer);
Matrix * predict(Matrix * input, NN * nn);

// Training Operations
Matrix * feed_forward(Matrix * input, NN * nn);
Matrix * feed_forward_internal(Matrix * input, Layer * current_layer);

void back_propagate(NN * nn, Matrix * output_error);
void update_weights(Layer * layer, Matrix * new_weights);
void update_bias(Layer * layer, Matrix * bias);
void fit (NN * nn, Dataset * dataset, int epochs);

// Display Operations
void display_node(Node * node);
void display_layer(Layer * layer);
void display_NN(NN * nn);
void display_weights(Layer * layer);
void display_NN_with_weights(NN * nn);
void display_datapoint(Data * data);
void display_dataset(Dataset * dataset);

// Free Operations
void free_NN(NN * nn);
void free_NN_layer(Layer * curr_layer);
void free_matrix(Matrix * matrix);
void free_inputs_outputs(Layer * input_layer);

// Activation functions
void relu(Matrix * matrix);
void relu_prime(Matrix * matrix);

// Cost functions
double mean_squared_error(Matrix * y_pred, Matrix * y_true);
Matrix * mean_squared_error_matrices(Matrix * y_pred, Matrix * y_true);
Matrix * loss_prime(Matrix * y_true, Matrix * y_pred);
double mean_squared_error_prime(double y_true, double y_pred);

// Dataset operations
Data * alloc_data();
Dataset * generate_not_dataset(int num_samples, int sample_length);
Matrix * generate_not_datapoint_x(int sample_length);
Matrix * generate_not_datapoint_y(Matrix * x);
void free_not_dataset(Dataset * dataset);

int main (void)
{
    srand(time(NULL));

    Dataset * auto_set = generate_not_dataset(100, 6);

    NN * nn = init_NN(0.001);
    add_layer(nn, 6);
    add_layer(nn, 40);
    add_layer(nn, 40);
    add_layer(nn, 40);
    add_layer(nn, 40);
    add_layer(nn, 6);

    fit (nn, auto_set, 100);

    Matrix * not_point = generate_not_datapoint_x(6);
    Matrix * output = predict(not_point, nn);
    printf("\n /|\\ Predictions /|\\\n");
    printf(" \\|/             \\|/\n");
    printf("\tInput:\n");
    display_matrix(not_point);
    printf("\tOutput:\n");
    display_matrix(output);

    free_NN(nn);
    nn = NULL;
    free_matrix(not_point);
    free_matrix(output);
    free_not_dataset(auto_set);

    return 0;
}

///                    ///
/// Dataset Operations ///
///                    ///

void free_not_dataset(Dataset * dataset) {
    for (int i = 0; i < dataset->num_examples; i++) {
        free_matrix(dataset->x[i]);
        free_matrix(dataset->y[i]);
    }
    free(dataset->x);
    free(dataset->y);
    free(dataset);
}

Matrix * generate_not_datapoint_x(int sample_length) {
    Matrix * not_x = alloc_matrix(1, sample_length);
    for (int i = 0; i < sample_length; i++)
        not_x->array[0][i] = rand() % 2;
    return not_x;
}

Matrix * generate_not_datapoint_y(Matrix * x) {
    Matrix * not_y = alloc_matrix(1, x->cols);
    for (int i = 0; i < x->cols; ++i)
        if (x->array[0][i] == 1)
            not_y->array[0][i] = 0;
        else
            not_y->array[0][i] = 1;
    return not_y;
}

// Generates x and y values in this relation:
/*
    x: [ 0 1 0 0 1 1 0 ]
    y: [ 1 0 1 1 0 0 1 ]
*/
Dataset * generate_not_dataset(int num_samples, int sample_length) {
    Dataset * dataset = (Dataset *) malloc(sizeof(Dataset) * 1);
    dataset->x = (Matrix **) malloc(sizeof(Matrix *) * num_samples);
    dataset->y = (Matrix **) malloc(sizeof(Matrix *) * num_samples);
    dataset->num_examples = num_samples;

    for (int i = 0; i < num_samples; i++) {
        dataset->x[i] = generate_not_datapoint_x(sample_length);
        dataset->y[i] = generate_not_datapoint_y(dataset->x[i]);
    }
    return dataset;
}

///                           ///
/// Neural Network Operations ///
///                           ///

// Predict given the input
Matrix * predict(Matrix * input, NN * nn) {
    Matrix * output = feed_forward(input, nn);
    free_inputs_outputs(nn->input_layer);
    return output;
}

void free_inputs_outputs(Layer * input_layer) {
    Layer * current_layer = input_layer;
    free_matrix(input_layer->input);
    while (current_layer->next_layer != NULL) {
        free_matrix(current_layer->output);
        current_layer = current_layer->next_layer;
    }
}

// Takes a Dataset struct to fit the model.
// This function is for supervised learning
// It is assumed that matching indices are correlated datapoints
void fit (NN * nn, Dataset * dataset, int epochs) {

    Layer * last_layer = nn->input_layer;
    while (last_layer->next_layer != NULL)
        last_layer = last_layer->next_layer;

    double err = 0;
    Matrix * output;
    Matrix * error;
    Matrix * datapoint_x;
    Matrix * datapoint_y;
    for (int i = 0; i < epochs; i++) {
        for (int j = 0; j < dataset->num_examples; j++) {
            datapoint_x = copy_matrix(dataset->x[j]);
            datapoint_y = copy_matrix(dataset->y[j]);
            output = feed_forward(datapoint_x, nn);
            err += mean_squared_error(output, datapoint_y);

            error = loss_prime(output, datapoint_y);
            back_propagate(nn, error);

            free_matrix(error);
            free_matrix(output);
            free_matrix(datapoint_x);
            free_matrix(datapoint_y);
        }
        err /= dataset->num_examples;
        printf("<-> Epoch: %4.d, Error: %lf <->\n", i + 1, err);
    }

}

// The classic feed_forward algorithm.
// Implemented iteratively
Matrix * feed_forward_internal(Matrix * input, Layer * input_layer) {
    Layer * current_layer = input_layer;
    Matrix * weights;
    Matrix * output;

    current_layer = input_layer;

    while (current_layer->next_layer != NULL) {

        current_layer->input = input;
        weights = convert_to_matrix(current_layer);
        output = dot_product(input, weights);
        plus_biases(output, current_layer->next_layer);
        relu(output);
        current_layer->output = output;
        input = output;
        free(weights->array);
        free(weights);

        current_layer = current_layer->next_layer;

    }
    return output;
}

// Update the weights to the new weights
void update_weights(Layer * layer, Matrix * new_weights) {
    int rows = new_weights->rows, cols = new_weights->cols;
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            layer->nodes[i]->weights[j] = new_weights->array[i][j];
}

// Update the bias to the new bias
void update_bias(Layer * layer, Matrix * bias) {
    int rows = bias->rows;
    for (int i = 0; i < rows; ++i)
        layer->nodes[i]->bias = bias->array[i][0];
}

// Classic backpropagation algorithm implemented iteratively
void back_propagate(NN * nn, Matrix * output_error) {
    Layer * last_layer = nn->input_layer;
    while (last_layer->next_layer != NULL)
        last_layer = last_layer->next_layer;

    Layer * current_layer = last_layer;

    Matrix * weights;
    Matrix * weights_T;
    Matrix * input_error;
    Matrix * bias;
    Matrix * input_T;
    Matrix * weights_error;
    Matrix * relu_output_error;
    Matrix * relu_input;
    Matrix * input;

    while (current_layer->prev_layer != NULL) {
        weights = convert_to_matrix(current_layer->prev_layer);
        weights_T = transpose(weights);

        input_error = dot_product(output_error, weights_T);
        bias = nodes_to_matrix(current_layer);

        input = current_layer->prev_layer->input;
        input_T = transpose(input);
        weights_error = dot_product(input_T, output_error);

        multiply_by(weights_error, nn->learning_rate);
        multiply_by(output_error, nn->learning_rate);

        element_wise_subtraction(weights, weights_error);
        element_wise_subtraction(bias, output_error);

        update_weights(current_layer->prev_layer, weights);
        update_bias(current_layer, bias);

        relu_output_error = input_error;
        relu_input = current_layer->prev_layer->input;
        relu_prime(relu_input);
        element_wise_multiplication(relu_input, relu_output_error);

        output_error = relu_input;
        current_layer = current_layer->prev_layer;

        // Free that memory!
        free_matrix(weights_T);
        free_matrix(input_T);
        free_matrix(input_error);
        free_matrix(bias);
        free_matrix(weights_error);
        free(weights->array);
        free(weights);
    }
    free_inputs_outputs(nn->input_layer);
}

///                              ///
/// Cost & Activation Operations ///
///                              ///

// Primeify the loss to tell each weight how to adjust
Matrix * loss_prime(Matrix * y_true, Matrix * y_pred) {
    Matrix * loss_matrix = alloc_matrix(y_true->rows, y_true->cols);
    for (int i = 0; i < loss_matrix->rows; i++)
        for (int j = 0; j < loss_matrix->cols; j++)
            loss_matrix->array[i][j] = mean_squared_error_prime(y_true->array[i][j], y_pred->array[i][j]);
    free_matrix(y_true);
    free_matrix(y_pred);
    return loss_matrix;
}

// The prime of x^2
double mean_squared_error_prime(double y_pred, double y_true) {
    return 2.0 *(y_pred - y_true);
}

// Get the mean squared error of both matrices
Matrix * mean_squared_error_matrices(Matrix * y_pred, Matrix * y_true) {
    Matrix * error = alloc_matrix(y_pred->rows, y_pred->cols);
    int rows = y_pred->rows, cols = y_pred->cols;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            error->array[i][j] = pow(y_pred->array[i][j] - y_true->array[i][j] , 2);
    return error;
}

// Update each element x of the matrix to be the
// slope of the relu function at x
void relu_prime(Matrix * matrix) {
    int rows = matrix->rows, cols = matrix->cols;
    double num;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            num = matrix->array[i][j];
            if (num <= 0)
                matrix->array[i][j] = 0;
            else
                matrix->array[i][j] = 1;
        }
    }

}

// Allocate memory for a datapoint
Data * alloc_data() {
    Data * data_point = (Data *) malloc(sizeof(Data) * 1);
    data_point->x = NULL;
    data_point->y = NULL;
    return data_point;
}

// Random input based on the architecture of the Neural Network
Matrix * rand_input(NN * nn) {
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

// Error defined as: (y_pred - y_true) ^ 2
double mean_squared_error(Matrix * y_pred, Matrix * y_true) {
    double mse = 0;
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

///                  ///
/// Layer Operations ///
///                  ///

// Connect the two layers with random weights.
// The randomization of weights is implicit because connecting
// layers only occurs during the creation of the network
void connect_layers(Layer * layer1, Layer * layer2) {
    layer1->next_layer = layer2;
    layer2->prev_layer = layer1;
    for (int i = 0; i < layer1->node_count; i++)
        layer1->nodes[i]->weights = init_weights(layer2->node_count);
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

///                   ///
/// Matrix operations ///
///                   ///

// Generates a random matrix given rows and columns
Matrix * rand_matrix(int rows, int cols) {
    Matrix * matrix = alloc_matrix(rows, cols);
    random_populate(matrix);
    return matrix;
}

// Determines if two matrices can be accessed element wise
int element_wiseable(Matrix * m1, Matrix * m2) {
    return ((m1->rows == m2->rows && m1->cols == m2->cols) || (m1->rows == m2->cols && m1->cols == m1->rows) || (m1->cols == m2->rows && m1->rows == m2->cols));
}

// Element-wise subtraction so long as either the rows equal the rows,
// or the rows equal the columns and vice versa
void element_wise_subtraction(Matrix * m1, Matrix * m2) {
    if (!element_wiseable(m1, m2)) {
        printf("Mismatching dimensions for inclusive element-wise subtraction!\n");
        printf("Matrix 1:\n");
        display_matrix(m1);
        printf("Matrix 2:\n");
        display_matrix(m2);
        exit(1);
    }

    int r1 = m1->rows, c1 = m1->cols, r2 = m2->rows, c2 = m2->cols;
    if (r1 == r2 && c1 == c2) {
        for (int i = 0; i < r1; ++i)
            for (int j = 0; j < c1; ++j)
                m1->array[i][j] -= m2->array[i][j];
    } else if (r1 == c2 && c1 == r2) {
        Matrix * m2_copy = copy_matrix(m2);
        Matrix * temp = transpose(m2_copy);
        element_wise_subtraction(m1, temp);
        free_matrix(m2_copy);
        free_matrix(temp);
    }
}

// Element-wise multiplication so long as either the rows equal the rows,
// or the rows equal the columns and vice versa
void element_wise_multiplication(Matrix * m1, Matrix * m2) {

    if (!element_wiseable(m1, m2)) {
        printf("Mismatching dimensions for element-wise multiplication!\n");
        printf("Matrix 1:\n");
        display_matrix(m1);
        printf("Matrix 2:\n");
        display_matrix(m2);
        exit(1);
    }

    int r1 = m1->rows, c1 = m1->cols, r2 = m2->rows, c2 = m2->cols;
    if (r1 == r2 && c1 == c2) {
        for (int i = 0; i < r1; ++i)
            for (int j = 0; j < c1; ++j)
                m1->array[i][j] *= m2->array[i][j];
    } else if (r1 == c2 && c1 == r2) {
        Matrix * m2_copy = copy_matrix(m2);
        Matrix * temp = transpose(m2_copy);
        element_wise_multiplication(m1, temp);
        free_matrix(m2_copy);
        free_matrix(temp);
    }
}


// Element-wise addition so long as either the rows equal the rows,
// or the rows equal the columns and vice versa
void element_wise_addition(Matrix * m1, Matrix * m2) {

    if (!element_wiseable(m1, m2)) {
        printf("Mismatching dimensions for element-wise addition!\n");
        printf("Matrix 1:\n");
        display_matrix(m1);
        printf("Matrix 2:\n");
        display_matrix(m2);
        exit(1);
    }

    int r1 = m1->rows, c1 = m1->cols, r2 = m2->rows, c2 = m2->cols;
    if (r1 == r2 && c1 == c2) {
        for (int i = 0; i < r1; ++i)
            for (int j = 0; j < c1; ++j)
                m1->array[i][j] += m2->array[i][j];
    } else if (r1 == c2 && c1 == r2) {
        Matrix * m2_copy = copy_matrix(m2);
        Matrix * temp = transpose(m2_copy);
        element_wise_addition(m1, temp);
        free_matrix(m2_copy);
        free_matrix(temp);
    }
}

// Element-wise division so long as either the rows equal the rows,
// or the rows equal the columns and vice versa
void element_wise_division(Matrix * m1, Matrix * m2) {
    if (!element_wiseable(m1, m2)) {
        printf("Mismatching dimensions for element-wise division!\n");
        printf("Matrix 1:\n");
        display_matrix(m1);
        printf("Matrix 2:\n");
        display_matrix(m2);
        exit(1);
    }

    int r1 = m1->rows, c1 = m1->cols, r2 = m2->rows, c2 = m2->cols;
    if (r1 == r2 && c1 == c2) {
        for (int i = 0; i < r1; ++i)
            for (int j = 0; j < c1; ++j)
                m1->array[i][j] /= m2->array[i][j];
    } else if (r1 == c2 && c1 == r2) {
        Matrix * m2_copy = copy_matrix(m2);
        Matrix * temp = transpose(m2_copy);
        element_wise_division(m1, temp);
        free_matrix(m2_copy);
        free_matrix(temp);
    }
}

// Multiply each element in a matrix by some number
void multiply_by (Matrix * matrix, double value) {
    for (int i = 0; i < matrix->rows; i++)
        for (int j = 0; j < matrix->cols; j++)
            matrix->array[i][j] *= value;
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
    Matrix * matrix = (Matrix *) malloc(sizeof(Matrix) * 1);
    matrix->rows = layer->node_count;
    matrix->cols = layer->next_layer->node_count;
    matrix->array = (double **) malloc(sizeof(double *) * matrix->rows);
    for (int i = 0; i < layer->node_count; i++)
        matrix->array[i] = layer->nodes[i]->weights;
    return matrix;
}

// Transposes the matrix then frees the original matrix
Matrix * transpose_destroy(Matrix * matrix) {
    Matrix * transposed_matrix = alloc_matrix(matrix->cols, matrix->rows);
    zero_init(transposed_matrix);
    int rows = transposed_matrix->rows, cols = transposed_matrix->cols;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            transposed_matrix->array[i][j] = matrix->array[j][i];

    free_matrix(matrix);
    return transposed_matrix;
}

// Transposes the matrix without freeing the original
Matrix * transpose(Matrix * matrix) {
    Matrix * transposed_matrix = alloc_matrix(matrix->cols, matrix->rows);
    zero_init(transposed_matrix);
    int rows = transposed_matrix->rows, cols = transposed_matrix->cols;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            transposed_matrix->array[i][j] = matrix->array[j][i];

    return transposed_matrix;
}

// Initializes the given matrix with 0s
void zero_init(Matrix * matrix) {
    for (int r = 0; r < matrix->rows; r++)
        for (int c = 0; c < matrix->cols; c++)
            matrix->array[r][c] = 0;
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

// Takes arrays with specified dimensions and returns the dot product (destroys the passed matrices)
Matrix * dot_product_destroy(Matrix * m1, Matrix * m2)
{
    // Check for mismatching dimensions
    if (m1->cols != m2->rows) {
        printf("Mismatching matrices dimensions!\n");
        printf("Matrix 1:\n");
        display_matrix(m1);
        printf("Matrix 2:\n");
        display_matrix(m2);
        exit(1);
    }

    Matrix * result_matrix = alloc_matrix(m1->rows, m2->cols);
    zero_init(result_matrix);
    for (int i = 0; i < m1->rows; ++i)
        for (int j = 0; j < m2->cols; ++j)
            for (int k = 0; k < m1->cols; ++k)
                result_matrix->array[i][j] += m1->array[i][k] * m2->array[k][j];
    free_matrix(m1);
    free_matrix(m2);
    return result_matrix;
}

// Takes arrays with specified dimensions and returns the dot product
Matrix * dot_product(Matrix * m1, Matrix * m2)
{
    // Check for mismatching dimensions
    if (m1->cols != m2->rows) {
        printf("Mismatching matrices dimensions!\n");
        printf("Matrix 1:\n");
        display_matrix(m1);
        printf("Matrix 2:\n");
        display_matrix(m2);
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

///                   ///
/// Memory Operations ///
///                   ///

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

// User friendly abstraction over free_NN_layer
void free_NN(NN * nn) {
    free_NN_layer(nn->input_layer);
    free(nn->input_layer);
    free(nn);
}

// Initialize a neural network struct
NN * init_NN(double learning_rate) {
    NN * nn = (NN *) malloc(sizeof(NN) * 1);
    nn->input_layer = NULL;
    nn->layers = 0;
    nn->learning_rate = learning_rate;
    return nn;
}

// Initializes and allocates a Node * struct
Node * init_node() {
    Node * new_node = (Node *) malloc(sizeof(Node) * 1);
    new_node->bias = rand_in_range(-0.1, 0.1);
    new_node->weights = NULL;
    return new_node;
}

// Frees the given array with the given dimensions
void free_matrix(Matrix * matrix) {
    for (int i = 0; i < matrix->rows; i++)
        free(matrix->array[i]);

    free(matrix->array);
    free(matrix);
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

// Initializes an array of random values
double * init_weights(int nodes) {
    double * weights = (double *) malloc(sizeof(double) * nodes);
    for (int i = 0; i < nodes; i++)
        weights[i] = rand_in_range(-0.5, 0.5);
    return weights;
}

// Initializes and allocates a Layer * struct given the number of nodes
Layer * init_layer(int node_count) {
    Layer * new_layer = (Layer *) malloc(sizeof(Layer) * 1);
    new_layer->next_layer = NULL;
    new_layer->prev_layer = NULL;
    new_layer->node_count = node_count;
    new_layer->nodes = (Node **) malloc(sizeof(Node *) * node_count);
    new_layer->input = NULL;
    new_layer->output = NULL;
    for (int i = 0; i < node_count; i++)
        new_layer->nodes[i] = init_node();
    return new_layer;
}

///                    ///
/// Display Operations ///
///                    ///

// Print the shape of the given matrix
void print_shape(Matrix * matrix) {
    printf("(%d, %d)\n", matrix->rows, matrix->cols);
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

// Displays the matrix given the dimensions
void display_matrix(Matrix * matrix) {
   double num = 0.0;
   if (matrix == NULL)
        return;
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

// Display the given dataset
void display_dataset(Dataset * dataset) {
    printf("Dataset of %d examples\n", dataset->num_examples);
    for (int i = 0; i < dataset->num_examples; i++) {
        printf("X%d:\n", i);
        display_matrix(dataset->x[i]);
        printf("Y%d:\n", i);
        display_matrix(dataset->y[i]);
    }
}

// Display one point of data
void display_datapoint(Data * data) {
    printf("X:\n");
    display_matrix(data->x);
    printf("Y:\n");
    display_matrix(data->y);
}
