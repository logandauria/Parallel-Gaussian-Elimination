/**
 * Gaussian Elimination implemented normally and in parallel using threads.
 *
 * @author Logan D'Auria
 *         lxd1644@rit.edu
 */
import java.io.*;
import java.util.ArrayList;

public class GaussianElimination {

    // set to true to use a designated file (next variable) for the algorithm
    private static final boolean fileInput = true;

    // desired file to obtain matrix information from
    private static final String FILENAME = "256.txt";

    // whether or not to print all matrix information and solutions before and after elimination
    private static boolean PRINT_FLAG = true;

    // set to true to run random tests through the algorithm
    private static final boolean randomTests = true;

    // ONLY USED FOR RANDOM GENERATION: desired matrix size for random generation
    private static final int N = 512;

    // range of numbers generated in the random number matrix (0 to VALUE_RANGE)
    private static final int VALUE_RANGE = 10;

    /**
     * Performs basic swapping of two given rows of the given matrix
     * @return new matrix with swapped rows
     */
    public static double[][] swapRow(double[][] matrix, int x, int y) {
        for (int k = 0; k <= matrix.length; k++) {
            double temp = matrix[x][k];
            matrix[x][k] = matrix[y][k];
            matrix[y][k] = temp;
        }
        return matrix;
    }

    /**
     * FOR TESTING PURPOSES
     * NON-PARALLEL Algorithm for gauss elimination used to compare against the parallel algorithm
     * @param matrix
     * @return a triangular matrix
     */
    public static double[][] gaussElim(double[][] matrix){
        int n = matrix.length;
        for (int k = 0; k < n; k++){
            int index = k;
            double value = matrix[index][k];

            // find row for pivot
            for (int x = k+1; x < n; x++) {
                if (Math.abs(matrix[x][k]) > value) {
                    value = matrix[x][k];
                    index = x;
                }
            }
            matrix = swapRow(matrix, k, index);

            // eliminate elements to form triangular matrix
            for (int x = k + 1; x < n; x++){
                double f = matrix[x][k] / matrix[k][k];
                for (int y = k + 1; y <= n; y++)
                    matrix[x][y] -= matrix[k][y]*f;
                matrix[x][k] = 0;
            }
        }
        return matrix;
    }

    /**
     * Uses back substitution to find the solution vector of the given matrix
     * Must be run serially
     * @param matrix
     * @return
     */
    public static double[] backSub(double matrix[][]) {

        int n = matrix.length;
        double sol[] = new double[n];

        // work up solutions from the bottom row with substitution
        for (int x = n - 1; x >= 0; x--){
            sol[x] = matrix[x][n];
            for (int y=x+1; y<n; y++){
                sol[x] = sol[x] - matrix[x][y] * sol[y];
            }
            sol[x] = sol[x] / matrix[x][x];
        }
        return sol;
    }

    /**
     * Prints a given matrix out, elements separated by spaces
     */
    public static void printMatrix(double[][] m){
        for(int x = 0; x < m.length; x++ ){
            for(int y = 0; y < m[0].length; y++){
                System.out.print(m[x][y] + " ");
            }
            System.out.println();
        }
    }

    /**
     * Randomly generates a new matrix with a given amount of rows and cols
     */
    public static double[][] randMatrix(int rows, int cols){
        double[][] matrix = new double[rows][cols];
        for(int x = 0; x < rows; x++){
            for(int y = 0; y < cols; y++){
                matrix[x][y] = (int)(Math.random() * VALUE_RANGE);
            }
        }
        return matrix;
    }

    /**
     * Verifies the solution vector of a given matrix
     * @return true if the solutions add up properly
     */
    public static boolean checkSolution(double[][] matrix, double[] sol) {
        int n = matrix[0].length;
        for(int x = 0; x < matrix.length; x++){
            double sum = 0;
            for(int y = 0; y < n - 1; y++){
                sum += matrix[x][y]*sol[y];
            }
            //checks that solution is approximately equal since program deals with minute decimals
            if(sum - matrix[x][n - 1] <= -0.1 || sum - matrix[x][n - 1] >= 0.1){
                return false;
            }
        }
        return true;
    }

    /**
     * Reads a file with a given matrix and turns it into a 2d array of doubles
     * @preconditon first line of file is matrix size followed by numbers
     * separated by spaces and endlines
     * @return matrix read from file
     */
    public static double[][] readFileMatrix() throws IOException {

        File file = new File(FILENAME);
        BufferedReader read = new BufferedReader(new FileReader(file));

        int n = Integer.parseInt(read.readLine().trim());
        double[][] matrix = new double[n][n + 1];

        String[] s;
        for (int x = 0; x < n; x++) {
            s = read.readLine().split(" ");
            for (int y = 0; y < n + 1; y++) {
                matrix[x][y] = Double.parseDouble(s[y]);
            }
        }

        return matrix;
    }

    /**
     * FOR TESTING PURPOSES
     * Computes and times the gaussian algorithm without parallelization
     */
    public static void normalGauss(double[][] matrix, int n){

        if(PRINT_FLAG) printMatrix(matrix);

        final double startTime = System.nanoTime();
        double[][] M = gaussElim(matrix);
        double[] solution = backSub(matrix);

        final double duration = (System.nanoTime() - startTime) / 1000000;

        if(PRINT_FLAG) {
            System.out.println("Triangular matrix:\n");
            printMatrix(matrix);
            System.out.println("\nSolutions:");
            for(int x = 0; x < solution.length; x++){
                System.out.println(solution[x]);
            }
        }
        System.out.println("\n" + duration + "ms passed");
    }

    /**
     * Computes and times the gaussian algorithm WITH parallelization
     * Each row of the matrix represents a task. The algorithm splits up the tasks
     * based on the amount of available processors and creates new threads to
     * complete equal amounts of tasks
     */
    public static void parallelGauss(double[][] matrix, int n){

        if(PRINT_FLAG) printMatrix(matrix);

        final double startTime = System.nanoTime();         // tracks the runtime of the algorithm
        int p = Runtime.getRuntime().availableProcessors(); // counts available processors
        int div = n / p;                                    // amount of tasks per threaed
        int remainder = n % p;                              // accounts for leftover tasks if no even split

        ArrayList<Worker> threads = new ArrayList<Worker>();// list of the worker threads
        for(int k = 0; k < p; k++){
            if(k == p - 1)
                threads.add(new Worker(matrix, k * div, (k+1) * div + remainder));  // accounts for leftover tasks
            else
                threads.add(new Worker(matrix, k * div, (k+1) * div));
            threads.get(k).run();
        }
        double[] solution = backSub(matrix);
        final double duration = (System.nanoTime() - startTime) / 1000000; // converts to ms

        // prints matrix solving information
        if(PRINT_FLAG) {
            System.out.println("Triangular matrix:\n");
            printMatrix(matrix);
            System.out.println("\nSolutions:");
            for(int x = 0; x < solution.length; x++){
                System.out.println(solution[x]);
            }
        }
        System.out.println("\n" + duration + "ms passed");
        if(checkSolution(matrix, solution)) System.out.println("solution is correct\n");

    }

    public static void main(String[] args) {

        // Reads a file into a 2d matrix array and runs algorithm on it
        if(fileInput) {
            try{
                double[][] matrix = readFileMatrix();
                parallelGauss(matrix, matrix.length);
            } catch (IOException e){
                System.err.println("Error reading file");
            }
        }

        // Conducts random tests for different matrix sizes
        if(randomTests){
            PRINT_FLAG = false;
            System.out.println("Testing parallel algorithm 10 times for random size 64 matrix: ");
            for(int x = 0; x < 10; x++) {
                int n = 64;
                double[][] matrix = randMatrix(n, n + 1);
                parallelGauss(matrix, n);
            }
            System.out.println("Testing parallel algorithm 10 times for random size 256 matrix: ");
            for(int x = 0; x < 10; x++) {
                int n = 256;
                double[][] matrix = randMatrix(n, n + 1);
                parallelGauss(matrix, n);
            }
            System.out.println("Testing parallel algorithm 10 times for random size 512 matrix: ");
            for(int x = 0; x < 10; x++) {
                int n = 512;
                double[][] matrix = randMatrix(n, n + 1);
                parallelGauss(matrix, n);
            }
            System.out.println("Testing parallel algorithm 10 times for random size 1024 matrix: ");
            for(int x = 0; x < 10; x++) {
                int n = 1024;
                double[][] matrix = randMatrix(n, n + 1);
                parallelGauss(matrix, n);
            }
//            System.out.println("Testing parallel algorithm 10 times for random size 4096 matrix: ");
//            for(int x = 0; x < 10; x++) {
//                int n = 4096;
//                double[][] matrix = randMatrix(n, n + 1);
//                parallelGauss(matrix, n);
//            }
        }
    }
}