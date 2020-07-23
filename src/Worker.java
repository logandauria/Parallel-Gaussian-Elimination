/**
 * Worker threads to assist in the Parallelization of the gaussian elimination algorithm
 *
 * @author Logan D'Auria
 *         lxd1644@rit.edu
 */
public class Worker extends Thread {

    private double[][] matrix;
    private int k;
    private int e;

    /**
     * Initiate the worker thread with range of tasks for parallel computation
     * @param matrix the matrix being processed by gaussian elimination
     * @param k lower bound of the range of rows to process by thread
     * @param e upper bound of the range of rows to process by thread
     */
    public Worker(double[][] matrix, int k, int e) {
        this.matrix = matrix;
        this.k = k;
        this.e = e;
    }

    /**
     *  Thread algorithm for gaussian elimination. Uses variables
     *  k and e to process a specific part of the matrix
     */
    public void run() {

        int n = matrix.length;

        for(;k < e; k++){
            int index = k;
            double value = matrix[index][k];

            // find row to pivot
            for (int x = k + 1; x < n; x++) {
                if (Math.abs(matrix[x][k]) > value) {
                    value = matrix[x][k];
                    index = x;
                }
            }
            // swap row and avoid self swap
            if (k != index) {
                matrix = GaussianElimination.swapRow(matrix, k, index);
            }

            // eliminate elements to form triangular matrix
            for (int x = k + 1; x < n; x++) {
                double f = matrix[x][k] / matrix[k][k];
                for (int y = k + 1; y <= n; y++)
                    matrix[x][y] -= matrix[k][y] * f;
                // fill in triangular zeros
                matrix[x][k] = 0;
            }
        }
    }
}
