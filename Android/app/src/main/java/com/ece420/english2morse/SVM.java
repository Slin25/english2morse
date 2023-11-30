package com.ece420.english2morse;


import android.util.Log;

import org.opencv.core.Mat;
import org.opencv.core.Core;
import java.io.InputStream;
import java.util.List;
import org.opencv.core.CvType;

public class SVM {
    private List raw_svm_weights;
    private Mat theta = new Mat(26, (28 * 28 + 1), CvType.CV_32FC1);


    public SVM(InputStream file) {
        CSVFile csv_file = new CSVFile(file);
        raw_svm_weights = csv_file.read();

        String[] l;

        for (int i = 0; i < raw_svm_weights.size(); i++) {
            l = (String[]) raw_svm_weights.get(i);

            for (int j = 0; j < l.length; j++) {
                // Parsing string to double
                theta.put(i,j,Double.parseDouble(l[j]));
            }
        }

    }

    public String predict(Mat resized_x) {
        Mat x_val = new Mat(28*28+1,1, CvType.CV_32FC1);

        for (int i = 0; i < x_val.rows(); i++) {
            if(i==0)
                x_val.put(i, 0, 1);
            else
                x_val.put(i, 0, resized_x.get(i-1, 0));
        }

        // Calculate the probability vector
        Mat yMat = new Mat(26,1, CvType.CV_32FC1);
        float[] y_hat = new float[26];

        // compute yMat = theta * x_val
        Core.gemm(theta, x_val, 1.0, new Mat(), 0.0, yMat, 0);

        // Copy yMat to y_hat
        yMat.get(0,0, y_hat);

        // Find max probability idx
        int largest_idx = 0;

        for (int i = 1; i < y_hat.length; i++) {
            if (y_hat[largest_idx] < y_hat[i]) {
                largest_idx = i;
            }
            Log.e("PREDICT", i + ": " + y_hat[i]);
        }
        Log.e("PREDICT", "Largest idx " + largest_idx + ": " + y_hat[largest_idx]);
        return Character.toString((char)(largest_idx + 1 + 64));
    }
}
