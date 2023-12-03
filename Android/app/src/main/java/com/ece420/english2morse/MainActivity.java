package com.ece420.english2morse;

import android.content.pm.ActivityInfo;
import android.content.res.AssetManager;
import android.graphics.BitmapFactory;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.PixelFormat;
import android.graphics.Rect;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
// import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.util.Log;
import android.widget.CompoundButton;
import android.widget.Switch;
import android.widget.TextView;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.nio.MappedByteBuffer;
import android.content.res.AssetFileDescriptor;
import java.io.FileInputStream;
import java.nio.channels.FileChannel;
import java.util.*;

import com.ece420.english2morse.ml.Emnist;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2  {

    // UI Variable
    private SurfaceView surfaceView;
    private SurfaceHolder surfaceHolder;

    private static final String TAG = "MainActivity";
    private Button captureButton;
    private Button playButton;
    private TextView predicted_String;
    private TextView morse_String;
    private Switch model_switch;
    // Camera Variable
    private CameraBridgeViewBase mOpenCvCameraView;

    private int my_width;
    private int my_height;

    // Mat to store RGBA and Grayscale camera preview frame
    private Mat mRgba;
    private Mat mGray;
    private Mat mCrop;

    private Rect2d myROI = new Rect2d(0,0,0,0);
    private int myROIWidth = 520;
    private int myROIHeight = 280;
    private Scalar myROIColor = new Scalar(0,100,0);

    private EdgeDetection edge_d = new EdgeDetection(myROIWidth, myROIHeight);
    private Mat edge_mCrop;
    private SVM my_svm;
    private InputStream file;
    private String pred_letter;
    private String morse_translation;
    private double[] retEdge;
    private List<Mat> retTextSeg;
    MediaPlayer mediaPlayer = new MediaPlayer();

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
        morse_String = (TextView) findViewById(R.id.morse);
        surfaceView = (SurfaceView)findViewById(R.id.debugView);
        surfaceHolder = surfaceView.getHolder();
        // Setup Button for Capture and playing audio
        captureButton = (Button) findViewById(R.id.captureBtn);
        playButton = (Button) findViewById(R.id.playBtn);
        // Setup Switch for choosing model, True = tf SVM model, False = regular SVM model
        model_switch = (Switch) findViewById(R.id.modelSwitch);
        model_switch.setChecked(true);

        playButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String tempFile = "";
                for (int i = 0; i < morse_translation.length(); i++) {
                    tempFile = Constants.audioFileMap.get(morse_translation.charAt(i));
                    if (tempFile != null) {
                        playSound(tempFile);
                    }
                }
            }
        });

        captureButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mCrop = mRgba.submat((int) (myROI.y),(int) (myROI.y+myROIHeight),(int) (myROI.x), (int) (myROI.x+myROIWidth));

//                drawSurface(surfaceHolder);
                english2Morse();
            }
        });

        // Setup OpenCV Camera View
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.originCam);
        // Use main camera with 0 or front camera with 1
        mOpenCvCameraView.setCameraIndex(0);
        // Force camera resolution, ignored since OpenCV automatically select best ones
        mOpenCvCameraView.setCvCameraViewListener(this);

    }

    public void newClassify(float [] image) {
        try {
            Emnist model = Emnist.newInstance(getApplicationContext());
            ByteBuffer byteBuffer = ByteBuffer.allocate(4 * 28 * 28);
            byteBuffer.order(ByteOrder.nativeOrder());
            for (int i = 0; i < 28 * 28; i++) {
                byteBuffer.putFloat(image[i]);
            }

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 28, 28, 1}, DataType.FLOAT32);
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Emnist.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            float[] output = outputFeature0.getFloatArray();

            int largest_idx = 0;
            for (int i = 1; i < output.length; i++) {
                Log.e("newML", "" + Character.toString((char)(i + 1 + 64)) + " " + output[i]);
                if (output[largest_idx] < output[i]) {
                    largest_idx = i;
                }
            }
            pred_letter = Character.toString((char)(largest_idx + 64));

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }

    public float[] getImageFromAssets(String fileName) throws IOException {
        AssetManager assetManager = getAssets();

        InputStream istr = assetManager.open(fileName);
        Bitmap bitmap = BitmapFactory.decodeStream(istr);
        istr.close();
        int [] tFile = new int[bitmap.getWidth()*bitmap.getHeight()];
        bitmap.getPixels(tFile, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        float [] t_float = new float[bitmap.getWidth()*bitmap.getHeight()];
        for (int i = 0; i < tFile.length; i++) {
            int pixel = ((tFile[i] >> 0) & 0xFF);
            float pixelf = (float) pixel;
            t_float[i] = pixelf;
        }
        Canvas canvas = surfaceHolder.lockCanvas(null);
        Matrix matrix = new Matrix();
        bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

        canvas.drawBitmap(bitmap, new Rect(0, 0, 28, 28), new Rect(0, 0, canvas.getWidth(), canvas.getHeight()), null);
        surfaceHolder.unlockCanvasAndPost(canvas);
        return t_float;
    }
    private MappedByteBuffer loadFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("emnist_94.tflite");
        FileInputStream inputStream=new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startOffset=fileDescriptor.getStartOffset();
        long declareLength=fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declareLength);
    }

    public String classifyImage(float [] flat_buf){
//        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 28, 28, 1}, DataType.FLOAT32);
//        inputFeature0.loadBuffer(bytebuffer);
        float[][][][] in_buf = new float[1][28][28][1];
        float[][] y_hat = new float[1][47];
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                in_buf[0][i][j][0] = flat_buf[i * 28 + j];
//                Log.e("DATA", "" + i + " " + in_buf[0][i][j][0]);
            }
        }
        try {
            Interpreter tflite = new Interpreter(loadFile());
            tflite.run(in_buf, y_hat);
            tflite.close();
            int largest_idx = 0;
            for (int i = 1; i < y_hat[0].length; i++) {
//                Log.e("ML", "" + Character.toString((char)(i + 1 + 64)) + " " + y_hat[0][i]);
                if (y_hat[0][largest_idx] < y_hat[0][i]) {
                    largest_idx = i;
                }
            }
            // Different indices for capital and lowercase letters
            if (largest_idx > 26) {
                largest_idx -= 27;
            }
            return Character.toString((char)(largest_idx + 64));
        } catch (IOException e){

        }
        return "";
    }
    public void english2Morse() {
        pred_letter = "";
//        Mat ret_ts = edge_d.drawTextSeg(mCrop);
//        draw(surfaceHolder, ret_ts);
//        drawLetter(surfaceHolder, edge_d.convertCameraImage(ret_ts));

        if (model_switch.isChecked()) {
            // Use tf SVM model

            // Text Segmentation
            retTextSeg = edge_d.performTextSeg(mCrop);
            float[] model_input_d = new float[28*28];
            int [] draw_last_letter = new int[28*28];
            for (int i = 0; i < retTextSeg.size(); i++) {
                Mat temp = retTextSeg.get(i);
                for (int j = 0; j < temp.rows(); j++) {
                    model_input_d[j] = (float) temp.get(j, 0)[0];
                    if (i == retTextSeg.size() -  1) {
                        int p = (int) temp.get(j, 0)[0];
                        draw_last_letter[j] = (int)(0xff000000 | p<<16 | p<<8 | p);
                    }
                }
                pred_letter += classifyImage(model_input_d);
            }
            retTextSeg.clear();

            // One letter
//            retEdge = edge_d.processImage(mCrop);
//            float[] model_input_d = new float[retEdge.length];
//            for (int i = 0; i < retEdge.length; i++) {
//                model_input_d[i] = (float) retEdge[i];
////            Log.e("Pixel", "" + i + " : " + model_input_d[i]);
//            }
//            pred_letter = classifyImage(model_input_d);


//            drawLetter(surfaceHolder, draw_last_letter);
//            drawSurface(surfaceHolder);
        } else {

            // Text Segmentation
            retTextSeg = edge_d.performTextSeg(mCrop);
            for (int i = 0; i < retTextSeg.size(); i++) {
                pred_letter += my_svm.predict(retTextSeg.get(i));
            }
            retTextSeg.clear();

            // One letter
//            retEdge = edge_d.processImage(mCrop);
//            Mat temp = new Mat(28,28, CvType.CV_32FC1);
//            temp.put(0, 0, retEdge);
//            temp = temp.reshape(0,28*28);
//            pred_letter = my_svm.predict(temp);

        }
        predicted_String.setText("PREDICTION: " + pred_letter);
        convertEnglishToMorse(pred_letter);
        morse_String.setText("MORSE CODE: " + morse_translation);
    }

    public void drawSurface(SurfaceHolder holder) {
        Canvas canvas = surfaceHolder.lockCanvas(null);
        Matrix matrix = new Matrix();
        int[] intArray = edge_d.convertCameraImage(mCrop);

        Bitmap bmp = Bitmap.createBitmap(intArray, 28, 28, Bitmap.Config.ARGB_8888);
        bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(), bmp.getHeight(), matrix, true);
//        canvas.drawBitmap(bmp, new Rect(0, 0, myROIWidth, myROIHeight), new Rect(0, 0, canvas.getWidth(), canvas.getHeight()), null);
        canvas.drawBitmap(bmp, new Rect(0, 0, 28, 28), new Rect(0, 0, canvas.getWidth(), canvas.getHeight()), null);
        surfaceHolder.unlockCanvasAndPost(canvas);
    }

    public void drawLetter(SurfaceHolder holder, int[] letter) {
        Canvas canvas = surfaceHolder.lockCanvas(null);
        Matrix matrix = new Matrix();

        Bitmap bmp = Bitmap.createBitmap(letter, 28, 28, Bitmap.Config.ARGB_8888);
        bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(), bmp.getHeight(), matrix, true);
        canvas.drawBitmap(bmp, new Rect(0, 0, 28, 28), new Rect(0, 0, canvas.getWidth(), canvas.getHeight()), null);
        surfaceHolder.unlockCanvasAndPost(canvas);
    }

    public void draw(SurfaceHolder holder, Mat m) {
        Canvas canvas = surfaceHolder.lockCanvas(null);
        Bitmap bmp = Bitmap.createBitmap(m.cols(), m.rows(),Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(m, bmp);
        canvas.drawBitmap(bmp, null, new Rect(0, 0, canvas.getWidth(), canvas.getHeight()), null);
        surfaceHolder.unlockCanvasAndPost(canvas);
    }

    public void playSound(String soundFileName) {
        try {
            AssetFileDescriptor descriptor = getAssets().openFd(soundFileName);
            mediaPlayer.reset();
            mediaPlayer.setDataSource(descriptor.getFileDescriptor(), descriptor.getStartOffset(), descriptor.getLength());
            descriptor.close();

            mediaPlayer.prepare();
            mediaPlayer.setLooping(false);
            mediaPlayer.start();
            Thread.sleep(1000);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    public void convertEnglishToMorse(String english_text) {
        morse_translation = "";
        for (int i = 0; i < english_text.length(); i++) {
            morse_translation += Constants.morseMap.get(english_text.charAt(i));
        }
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
        file = getResources().openRawResource(R.raw.model67);
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
//        Imgproc.rectangle(mRgba,
//                new Point(myROI.x, myROI.y),
//                new Point(myROI.x + myROI.width, myROI.y + myROI.height),
//                myROIColor,
//                4);

        Imgproc.rectangle(mRgba,
                new Point(myROI.x, myROI.y),
                new Point(myROI.x + myROIWidth, myROI.y + myROIHeight),
                myROIColor,
                4);

        // Returned frame will be displayed on the screen
        return mRgba;
    }
}
