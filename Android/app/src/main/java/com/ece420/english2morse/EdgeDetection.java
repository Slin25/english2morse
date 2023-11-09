package com.ece420.english2morse;

import java.lang.Math;

public class EdgeDetection  {
    private int width;
    private int height;
    // Kernels
    private double[][] kernelX = new double[][] {{1,0,-1},{1,0,-1},{1,0,-1}};
    private double[][] kernelY = new double[][] {{1,1,1},{0,0,0},{-1,-1,-1}};

    public EdgeDetection(int _width, int _height) {
        width = _width;
        height = _height;
    }

    // Helper function to merge the results and convert GrayScale to RGB
    public int[] merge(int[] xdata,int[] ydata){
        int size = height * width;
        int[] mergeData = new int[size];
        for(int i=0; i<size; i++)
        {
            int p = (int)Math.sqrt((xdata[i] * xdata[i] + ydata[i] * ydata[i]) / 2);
            mergeData[i] = 0xff000000 | p<<16 | p<<8 | p;
        }
        return mergeData;
    }

    public int[] conv2(byte[] data, int width, int height, double kernel[][]){
        // 0 is black and 255 is white.
        int size = height * width;
        int[] convData = new int[size];

        double flipped_kernel [][] = new double[kernel.length][kernel[0].length];

        for (int x = 0; x < kernel.length; x++) {
            for (int y = 0; y < kernel[0].length; y++) {
                flipped_kernel[kernel.length - 1 - x][kernel[0].length - 1 - y] = kernel[x][y];
            }
        }

        for (int x = 0; x < height; x++) {
            for (int y = 0; y < width; y++){
                for (int x_k = 0; x_k < flipped_kernel.length; x_k++) {
                    for (int y_k = 0; y_k < flipped_kernel[0].length; y_k++) {
                        int data_x = x - (int)((flipped_kernel.length - 1) / 2) + x_k;
                        int data_y = y - (int)((flipped_kernel[0].length - 1) / 2) + y_k;
                        if (data_x > 0 && data_x < height && data_y > 0 && data_y < width) {
                            convData[(x * width) + y] += (int) (flipped_kernel[x_k][y_k] * (data[(data_x * width) + data_y] & 0x00FF));
                        }

                    }
                }

            }
        }
        return convData;
    }

    public int[] performEdgeD(byte[] data) {
        int[] xData = conv2(data, width, height, kernelX);
        int[] yData = conv2(data, width, height, kernelY);

        return merge(xData, yData);
    }
}
