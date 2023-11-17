package com.ece420.english2morse;

import android.content.Intent;
import android.content.Context;
import android.content.pm.ActivityInfo;
import android.hardware.Camera;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.PixelFormat;
import android.graphics.Rect;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
// import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.util.Log;
import android.widget.TextView;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import java.nio.charset.StandardCharsets;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2  {

    // UI Variable
    private SurfaceView surfaceView;
    private SurfaceHolder surfaceHolder;

    private static final String TAG = "MainActivity";
    private Button captureButton;
    private TextView predicted_String;
    // Camera Variable
    private CameraBridgeViewBase mOpenCvCameraView;

    private int my_width;
    private int my_height;
    private int capture = 0;

    // Mat to store RGBA and Grayscale camera preview frame
    private Mat mRgba;
    private Mat mGray;
    private Mat mCrop;

    private Rect2d myROI = new Rect2d(0,0,0,0);
    private int myROIWidth = 600;
    private int myROIHeight = 400;
    private Scalar myROIColor = new Scalar(0,100,0);

    private EdgeDetection edge_d = new EdgeDetection(myROIWidth, myROIHeight);
    private Mat edge_mCrop;
    private SVM my_svm;
    private InputStream file;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setFormat(PixelFormat.UNKNOWN);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_camera);
        super.setRequestedOrientation (ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);

        if (!OpenCVLoader.initDebug()) {
            Log.e(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), not working.");
        } else {
            Log.d(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), working.");
        }

        predicted_String = (TextView) findViewById(R.id.predString);
        surfaceView = (SurfaceView)findViewById(R.id.debugView);
        surfaceHolder = surfaceView.getHolder();
        // Setup Button for Capture
        captureButton = (Button) findViewById(R.id.captureBtn);

        captureButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (capture == 1) {
                    captureButton.setText("CAPTURE");
                    drawSurface(surfaceHolder);
                    english2Morse();
                    capture = 0;
                } else {
                    captureButton.setText("NEW");
                    predicted_String.setText("NOTHING TO PREDICT");
                    capture = 1;
                }
            }
        });

        // Setup OpenCV Camera View
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.originCam);
        // Use main camera with 0 or front camera with 1
        mOpenCvCameraView.setCameraIndex(0);
        // Force camera resolution, ignored since OpenCV automatically select best ones
        mOpenCvCameraView.setCvCameraViewListener(this);

    }

    public void english2Morse() {
        Mat mResized = new Mat(28, 28, CvType.CV_8UC1);
        Imgproc.resize(edge_mCrop, mResized, mResized.size(), 0,0, Imgproc.INTER_AREA);

        // Vectorize the image
        mResized = mResized.reshape(0,28*28);
        mResized.convertTo(mResized,CvType.CV_32FC1);

        double Min = Core.minMaxLoc(mResized).minVal;
        double Max = Core.minMaxLoc(mResized).maxVal;
        double[] val1;
        for (int i = 0; i < mResized.rows() ; i++) {
            val1 = mResized.get(i,0);
            val1[0] = Math.floor((val1[0]-Min)*255/(Max-Min));
            // if needed, create negative image to make black background & white digit
            mResized.put(i,0,255 - val1[0]);
        }
        predicted_String.setText("PREDICTION: " + my_svm.predict(mResized));
    }

    public void drawSurface(SurfaceHolder holder) {
        Canvas canvas = surfaceHolder.lockCanvas(null);
        Matrix matrix = new Matrix();
//            matrix.postRotate(90);
//        double retData[] = new double[myROIWidth * myROIHeight];

        mCrop = mRgba.submat((int) (myROI.y),(int) (myROI.y+myROIHeight),(int) (myROI.x), (int) (myROI.x+myROIWidth));
        double[] retData = edge_d.performEdgeD(mCrop);
        edge_mCrop.put(0, 0, retData);
        int[] intArray = new int[retData.length];

        for (int i = 0; i < intArray.length; ++i)
            intArray[i] = (int) retData[i];

//        List<double[]> retData = edge_d.performTextSeg(mCrop);
//        Log.e("drawSurf", "" + retData.size());


//        if (retData.size() > 0) {
//            int[] intArray = new int[retData.get(0).length];
//            Log.e("drawSurf_len", "" + retData.get(0).length);
//            for (int i = 0; i < intArray.length; ++i) {
//                intArray[i] = (int) retData.get(0)[i];
//            }
//            Bitmap bmp = Bitmap.createBitmap(intArray, 28, 28, Bitmap.Config.ARGB_8888);
//            bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(), bmp.getHeight(), matrix, true);
//            canvas.drawBitmap(bmp, new Rect(0, 0, 28, 28), new Rect(0, 0, canvas.getWidth(), canvas.getHeight()), null);
//        }
        Bitmap bmp = Bitmap.createBitmap(intArray, myROIWidth, myROIHeight, Bitmap.Config.ARGB_8888);
        bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(), bmp.getHeight(), matrix, true);
        canvas.drawBitmap(bmp, new Rect(0, 0, myROIWidth, myROIHeight), new Rect(0, 0, canvas.getWidth(), canvas.getHeight()), null);
        surfaceHolder.unlockCanvasAndPost(canvas);
    }
    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };


    // OpenCV Camera Functionality Code
    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
        my_width = width;
        my_height = height;
        myROI = new Rect2d(my_width / 2 - myROIWidth / 2,
                my_height / 2 - myROIHeight / 2,
                myROIWidth,
                myROIHeight);
        file = getResources().openRawResource(R.raw.coef68_linear);
        edge_mCrop = new Mat(myROIHeight, myROIWidth, CvType.CV_32FC1);
        my_svm = new SVM(file);

    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        // Grab camera frame in rgba and grayscale format
        mRgba = inputFrame.rgba();
        // Grab camera frame in gray format
        mGray = inputFrame.gray();

        // Draw a rectangle on to the current frame
        Imgproc.rectangle(mRgba,
                new Point(myROI.x, myROI.y),
                new Point(myROI.x + myROI.width, myROI.y + myROI.height),
                myROIColor,
                4);

        // Returned frame will be displayed on the screen
        return mRgba;
    }

}
