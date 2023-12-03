package com.ece420.english2morse;

import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.lang.Math;
import java.util.*;

public class EdgeDetection  {
    private int width;
    private int height;
    // Kernels
    private double[][] kernelX = new double[][] {{1,0,-1},{1,0,-1},{1,0,-1}};
    private double[][] kernelY = new double[][] {{1,1,1},{0,0,0},{-1,-1,-1}};

    private int thresh = 130;   // Threshold for grayscale
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
//            mergeData[i] = 0xff000000 | p<<16 | p<<8 | p;
            mergeData[i] = p;
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

    public int[] convertCameraImage(Mat data) {
        Mat mResized = new Mat(28, 28, CvType.CV_32FC1);
        Imgproc.resize(data, mResized, mResized.size(), 0,0, Imgproc.INTER_AREA);
        mResized = mResized.reshape(0,28*28);

        int[] ret_data = new int[(int) mResized.total()];

        double Min = mResized.get(0, 0)[0];
        double Max = mResized.get(0, 0)[0];
        for (int i = 0; i < mResized.rows(); i++) {
            double temp = mResized.get(i, 0)[0];
            if (temp > Max) {
                Max = temp;
            }

            if (temp < Min) {
                Min = temp;
            }
        }

        double[] val1;
        int p;
        for (int i = 0; i < mResized.rows(); i++) {
            val1 = mResized.get(i, 0);
            val1[0] = Math.floor((val1[0]-Min)*255/(Max-Min));
            p = (int) (val1[0]);
            // if (i < 28 || i % 28 == 0 || i % 28 == 27 || i > 755)
            if (i < 28 || i % 28 == 0 || i % 28 == 27 || i > 755) {
                p = (int) (Math.floor(((int)(mResized.get(1, 0)[0])-Min)*255/(Max-Min)));
            }
            ret_data[i] = (int) (0xff000000 | p<<16 | p<<8 | p);
        }

        return ret_data;
    }
    public double[] performEdgeD(Mat data) {
        int[] xData = conv2(data, width, height, kernelX);
        int[] yData = conv2(data, width, height, kernelY);

        double[] edge_data = merge(xData, yData);
        double[] ret_edge = new double[edge_data.length];
        double Min = edge_data[0];
        double Max = edge_data[0];
        for (int i = 0; i < edge_data.length; i++) {
            if (edge_data[i] < Min) {
                Min = edge_data[i];
            }

            if (edge_data[i] > Max) {
                Max = edge_data[i];
            }
        }

        // 0 to 255 range
        double p;
        for (int i = 0; i < edge_data.length; i++) {

            if (i < width * 5 || i % width < 5 || i % width > width - 5 || i > ((width * height) - (width * 5) - 1)) {
                // remove border edges
                p = (double) (Math.floor(((edge_data[1])-Min)*255/(Max-Min)));
            } else {
                p = Math.floor((edge_data[i]-Min)*255/(Max-Min));
            }

            if (p < thresh) {
                p = 0;  // Black
            } else {
                p = 255; // White
            }

            ret_edge[i] = p;
        }

        return ret_edge;
    }

    public double[] processImage(Mat data) {
        Mat mResized = new Mat(28, 28, CvType.CV_32FC1);
        Imgproc.resize(data, mResized, mResized.size(), 0,0, Imgproc.INTER_AREA);
        mResized = mResized.reshape(0,28*28);

        double[] ret_data = new double[(int)(mResized.total())];
        double Min = mResized.get(0, 0)[0];
        double Max = mResized.get(0, 0)[0];
        for (int i = 0; i < mResized.rows(); i++) {
            double temp = mResized.get(i, 0)[0];
            if (temp > Max) {
                Max = temp;
            }

            if (temp < Min) {
                Min = temp;
            }
        }

        double[] val1;
        int p;
        for (int i = 0; i < mResized.rows(); i++) {
            val1 = mResized.get(i, 0);
            val1[0] = Math.floor((val1[0]-Min)*255/(Max-Min));
//            Log.e("Val1", "" + i + " : " + val1[0]);
            p = (int) (val1[0]);
            if (i < 27 || i % 28 == 0 || i % 28 == 27 || i > 755) {
                // Remove borders
                p = (int) (Math.floor(((int)(mResized.get(1, 0)[0])-Min)*255/(Max-Min)));
            }
            ret_data[i] = p;
        }

        return ret_data;
    }

    public double[] convertGrayscaleDouble(Mat data) {
        double [] ret_d = new double[(int)data.total()];

        double Min = data.get(0, 0)[0];
        double Max = data.get(0, 0)[0];
        for (int i = 0; i < data.rows(); i++) {
            double temp = data.get(i, 0)[0];
            if (temp > Max) {
                Max = temp;
            }

            if (temp < Min) {
                Min = temp;
            }
        }

        double[] val1;
        int p;
        for (int i = 0; i < data.rows(); i++) {
            val1 = data.get(i, 0);
            val1[0] = 255 - Math.floor((val1[0]-Min)*255/(Max-Min));
            p = (int) (255 - val1[0]);
            if (i < 27 || i % 28 == 0 || i % 28 == 27 || i > 755) {
                // Remove borders
                p = (int) (Math.floor(((int)(data.get(1, 0)[0])-Min)*255/(Max-Min)));
            }
            ret_d[i] = p;
        }

        return ret_d;
    }
    public Mat convertGrayscaleRange(Mat data) {
        Mat ret_gs = new Mat(data.rows(), data.cols(), CvType.CV_32FC1);
        double [] ret_d = new double[(int)ret_gs.total()];

        double Min = data.get(0, 0)[0];
        double Max = data.get(0, 0)[0];
        for (int i = 0; i < data.rows(); i++) {
            double temp = data.get(i, 0)[0];
            if (temp > Max) {
                Max = temp;
            }

            if (temp < Min) {
                Min = temp;
            }
        }

        double[] val1;
        int p;
        for (int i = 0; i < data.rows(); i++) {
            val1 = data.get(i, 0);
            val1[0] = 255 - Math.floor((val1[0]-Min)*255/(Max-Min));
            p = (int) (255 - val1[0]);
            if (i < 27 || i % 28 == 0 || i % 28 == 27 || i > 755) {
                // Remove borders
                p = (int) (Math.floor(((int)(data.get(1, 0)[0])-Min)*255/(Max-Min)));
            }
            ret_d[i] = p;
        }

        ret_gs.put(0, 0, ret_d);

        return ret_gs;
    }

    public Mat drawTextSeg(Mat data) {
        double[] grayscale_edge = performEdgeD(data);
//        double [] grayscale_edge = convertGrayscaleDouble(data);
        /*
            Testing Contours
        */
        Mat binary = new Mat(data.rows(), data.cols(), CvType.CV_8UC1);
        binary.put(0, 0, grayscale_edge);

//        List<MatOfPoint> contours = new ArrayList<>();
//        Mat hierarchy = new Mat();
//
//        Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
//        Mat contourImg = new Mat(data.size(), data.type());
//        Log.e("Contour", "" + contours.size());
//        for (int i = 0; i < contours.size(); i++) {
//            Imgproc.drawContours(binary, contours, i, new Scalar(100, 0, 0), 2);
//        }
        /*
            Testing Contours
         */

        return binary;
    }
    public List<Mat> performTextSeg(Mat data) {
        List<Mat> retText = new ArrayList<Mat>();
        double[] grayscale_edge = performEdgeD(data);


        List<Integer> col_pairs = findCutOffColumns(grayscale_edge);

        Log.e("COLS INFO", "" + col_pairs.size());
        for (int i = 0; i < col_pairs.size(); i+=2) {
            Mat temp = data.submat(0, height, col_pairs.get(i), col_pairs.get(i+1));

            Mat mResized = new Mat(28, 28, CvType.CV_32FC1);
            Imgproc.resize(temp, mResized, mResized.size(), 0,0, Imgproc.INTER_AREA);
            mResized = mResized.reshape(0,28*28);

            retText.add(convertGrayscaleRange(mResized));
        }
        return retText;
    }

    public List<Integer> findCutOffColumns(double[] data) {
        List<Integer> col_pairs = new ArrayList<Integer>();

//        boolean saw_white = false;
        boolean white_col = false;
        boolean new_pair = true;
        int temp_begin = 0;
//        col_pairs.add(0);
//        Log.e("COLS", "new " + 0);
        for (int i = 1; i < width; i++) {
            white_col = false;
            for (int j = 0; j < height; j++) {
                if (data[(j * width) + i] == 255) {
                    white_col = true;
                    if (new_pair) {
                        if (i - 10 > 0) {
                            temp_begin = i - 10;
                        } else {
                            temp_begin = i;
                        }

                        new_pair = false;
                    }
                    break;
                }
            }

            if (!new_pair && !white_col) {
                if (i - temp_begin > 10) {
                    col_pairs.add(temp_begin);
//                    Log.e("COLS", "begin " + temp_begin);
                    if (i + 10 < width) {
                        col_pairs.add(i + 10);
//                        Log.e("COLS", "end + 10 " + (i + 10));
                    } else {
                        col_pairs.add(i);
//                        Log.e("COLS", "end " + i);
                    }
                }
                new_pair = true;
            }

        }

        if (col_pairs.size() % 2 != 0) {
            col_pairs.add(width - 1);
            Log.e("COLS", "FIN " + (width - 1));
        }
        return col_pairs;
    }
}
