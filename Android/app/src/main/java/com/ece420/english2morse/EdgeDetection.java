package com.ece420.english2morse;

import android.util.Log;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.lang.Math;
import java.util.*;

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
    public double[] merge(int[] xdata, int[] ydata){
        int size = height * width;
        double[] mergeData = new double[size];
        for(int i=0; i<size; i++)
        {
            int p = (int)Math.sqrt((xdata[i] * xdata[i] + ydata[i] * ydata[i]) / 2);
            mergeData[i] = 0xff000000 | p<<16 | p<<8 | p;
        }
        return mergeData;
    }

    public int[] conv2(Mat data, int width, int height, double kernel[][]){
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
//                            convData[(x * width) + y] += (int) (flipped_kernel[x_k][y_k] * (data[(data_x * width) + data_y] & 0x00FF));
                            convData[(x * width) + y] += (int) (flipped_kernel[x_k][y_k] * (data.get(data_x, data_y)[0]));
                        }

                    }
                }

            }
        }
        return convData;
    }

    public double[] performEdgeD(Mat data) {
        int[] xData = conv2(data, width, height, kernelX);
        int[] yData = conv2(data, width, height, kernelY);

        double[] edge_data = merge(xData, yData);
        // Remove edge borders
        double min_v = 214748364.0;
        double max_v = -214748364.0;

        for (int i = 0; i < edge_data.length; i++) {
            if (edge_data[i] < min_v) {
                min_v = edge_data[i];
            }

            if (edge_data[i] > max_v) {
                max_v = edge_data[i];
            }
        }
        Log.e("EDGE_MIN", "" + min_v);
        Log.e("EDGE_MAX", "" + max_v);
        // Remove Top & Bottom edges
        for (int j = 0; j < width; j++) {
            edge_data[j] = min_v;
            edge_data[((1) * width) + j] = min_v;
            edge_data[((2) * width) + j] = min_v;
            edge_data[((3) * width) + j] = min_v;
            edge_data[((4) * width) + j] = min_v;
            edge_data[((height - 5) * width) + j] = min_v;
            edge_data[((height - 4) * width) + j] = min_v;
            edge_data[((height - 3) * width) + j] = min_v;
            edge_data[((height - 2) * width) + j] = min_v;
            edge_data[((height - 1) * width) + j] = min_v;
        }

        // Remove Left & Right edges
        for (int i = 0; i < height; i++) {
            edge_data[i * width] = min_v;
            edge_data[(i * width) + 1] = min_v;
            edge_data[(i * width) + 2] = min_v;
            edge_data[(i * width) + 3] = min_v;
            edge_data[(i * width) + 4] = min_v;
            edge_data[(i * width) + width - 5] = min_v;
            edge_data[(i * width) + width - 4] = min_v;
            edge_data[(i * width) + width - 3] = min_v;
            edge_data[(i * width) + width - 2] = min_v;
            edge_data[(i * width) + width - 1] = min_v;
        }

        return edge_data;
    }

    public int[] findTopBottomRow(int[] data) {
        int[] top_bottom = {-1, -1};
        boolean top = false;
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                if (data[(i * width) + j] == 255 && j != 0 && !top) {
                    top_bottom[0] = j;
                    top = true;
                } else if (data[(i * width) + j] == 255 && j != height - 1) {
                    top_bottom[1] = j;
                }
            }
        }

        return top_bottom;
    }

    public List<Integer> findCutOffColumns(double[] data, double max_value) {
        List<Integer> col_pairs = new ArrayList<Integer>();

        boolean end = false;
        int prev_col = -2;
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                if (data[(j * width) + i] == max_value && i != 0 && i != width - 1) {
                    if (!end) {
                        col_pairs.add(i);
                        Log.e("COLS", ""+i);
                        end = true;
                    } else {
                        prev_col = i;
                    }
                    continue;
                }
            }
            if (end && prev_col == i - 1) {
                col_pairs.add(prev_col);
                end = false;
            }
        }

        if (col_pairs.size() % 2 != 0) {
            col_pairs.add(width - 1);
        }
        return col_pairs;
    }


    public List<double[]> performTextSeg(Mat data) {
        List<double[]> text = new ArrayList<double[]>();
        double[] edge_data = performEdgeD(data);

        // Remove edge borders
        double min_v = 214748364.0;
        double max_v = -214748364.0;

        for (int i = 0; i < edge_data.length; i++) {
            if (edge_data[i] < min_v) {
                min_v = edge_data[i];
            }

            if (edge_data[i] > max_v) {
                max_v = edge_data[i];
            }
        }
        Log.e("EDGE_MIN", "" + min_v);
        Log.e("EDGE_MAX", "" + max_v);
        // Remove Top & Bottom edges
        for (int j = 0; j < width; j++) {
            edge_data[j] = min_v;
            edge_data[((1) * width) + j] = min_v;
            edge_data[((2) * width) + j] = min_v;
            edge_data[((3) * width) + j] = min_v;
            edge_data[((4) * width) + j] = min_v;
            edge_data[((height - 5) * width) + j] = min_v;
            edge_data[((height - 4) * width) + j] = min_v;
            edge_data[((height - 3) * width) + j] = min_v;
            edge_data[((height - 2) * width) + j] = min_v;
            edge_data[((height - 1) * width) + j] = min_v;
        }

        // Remove Left & Right edges
        for (int i = 0; i < height; i++) {
            edge_data[i * width] = min_v;
            edge_data[(i * width) + 1] = min_v;
            edge_data[(i * width) + 2] = min_v;
            edge_data[(i * width) + 3] = min_v;
            edge_data[(i * width) + 4] = min_v;
            edge_data[(i * width) + width - 5] = min_v;
            edge_data[(i * width) + width - 4] = min_v;
            edge_data[(i * width) + width - 3] = min_v;
            edge_data[(i * width) + width - 2] = min_v;
            edge_data[(i * width) + width - 1] = min_v;
        }

        // Find cutoff columns
        List<Integer> cols = findCutOffColumns(edge_data, max_v);

        for (int i = 0; i < cols.size(); i+=2) {
            int start = cols.get(i);
            int end = cols.get(i+1);
            Log.e("COLS", "START: " + start + " END: " + end);
            int idx = 0;
            double[] temp = new double[(end-start+1)*height];
            Mat crop_t = new Mat(height, (end-start+1), CvType.CV_32FC1);
            for (int x = start; x < end+1; x++) {
                for (int j = 0; j < height; j++) {
                    temp[idx] = edge_data[(j*width) + x];
                    idx++;
                }
            }
            crop_t.put(0, 0, temp);
            Mat mResized = new Mat(28, 28, CvType.CV_8UC1);
            Imgproc.resize(crop_t, mResized, mResized.size(), 0,0, Imgproc.INTER_AREA);
            mResized = mResized.reshape(0,28*28);
            mResized.convertTo(mResized,CvType.CV_32FC1);
            Log.e("RESIZED", mResized.dump());
            double[] resized_t = new double[28*28];
            mResized.put(0, 0, resized_t);
            text.add(resized_t);
        }

        return text;
    }
}
