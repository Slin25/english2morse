
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.face;

import java.lang.String;
import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Algorithm;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.utils.Converters;

// C++: class FaceRecognizer
//javadoc: FaceRecognizer

public class FaceRecognizer extends Algorithm {

    protected FaceRecognizer(long addr) { super(addr); }


    //
    // C++:  String getLabelInfo(int label)
    //

    //javadoc: FaceRecognizer::getLabelInfo(label)
    public  String getLabelInfo(int label)
    {
        
        String retVal = getLabelInfo_0(nativeObj, label);
        
        return retVal;
    }


    //
    // C++:  int predict(Mat src)
    //

    //javadoc: FaceRecognizer::predict(src)
    public  int predict_label(Mat src)
    {
        
        int retVal = predict_label_0(nativeObj, src.nativeObj);
        
        return retVal;
    }


    //
    // C++:  vector_int getLabelsByString(String str)
    //

    //javadoc: FaceRecognizer::getLabelsByString(str)
    public  MatOfInt getLabelsByString(String str)
    {
        
        MatOfInt retVal = MatOfInt.fromNativeAddr(getLabelsByString_0(nativeObj, str));
        
        return retVal;
    }


    //
    // C++:  void predict(Mat src, Ptr_PredictCollector collector)
    //

    //javadoc: FaceRecognizer::predict(src, collector)
    public  void predict_collect(Mat src, PredictCollector collector)
    {
        
        predict_collect_0(nativeObj, src.nativeObj, collector.getNativeObjAddr());
        
        return;
    }


    //
    // C++:  void predict(Mat src, int& label, double& confidence)
    //

    //javadoc: FaceRecognizer::predict(src, label, confidence)
    public  void predict(Mat src, int[] label, double[] confidence)
    {
        double[] label_out = new double[1];
        double[] confidence_out = new double[1];
        predict_0(nativeObj, src.nativeObj, label_out, confidence_out);
        if(label!=null) label[0] = (int)label_out[0];
        if(confidence!=null) confidence[0] = (double)confidence_out[0];
        return;
    }


    //
    // C++:  void read(String filename)
    //

    //javadoc: FaceRecognizer::read(filename)
    public  void read(String filename)
    {
        
        read_0(nativeObj, filename);
        
        return;
    }


    //
    // C++:  void setLabelInfo(int label, String strInfo)
    //

    //javadoc: FaceRecognizer::setLabelInfo(label, strInfo)
    public  void setLabelInfo(int label, String strInfo)
    {
        
        setLabelInfo_0(nativeObj, label, strInfo);
        
        return;
    }


    //
    // C++:  void train(vector_Mat src, Mat labels)
    //

    //javadoc: FaceRecognizer::train(src, labels)
    public  void train(List<Mat> src, Mat labels)
    {
        Mat src_mat = Converters.vector_Mat_to_Mat(src);
        train_0(nativeObj, src_mat.nativeObj, labels.nativeObj);
        
        return;
    }


    //
    // C++:  void update(vector_Mat src, Mat labels)
    //

    //javadoc: FaceRecognizer::update(src, labels)
    public  void update(List<Mat> src, Mat labels)
    {
        Mat src_mat = Converters.vector_Mat_to_Mat(src);
        update_0(nativeObj, src_mat.nativeObj, labels.nativeObj);
        
        return;
    }


    //
    // C++:  void write(String filename)
    //

    //javadoc: FaceRecognizer::write(filename)
    public  void write(String filename)
    {
        
        write_0(nativeObj, filename);
        
        return;
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  String getLabelInfo(int label)
    private static native String getLabelInfo_0(long nativeObj, int label);

    // C++:  int predict(Mat src)
    private static native int predict_label_0(long nativeObj, long src_nativeObj);

    // C++:  vector_int getLabelsByString(String str)
    private static native long getLabelsByString_0(long nativeObj, String str);

    // C++:  void predict(Mat src, Ptr_PredictCollector collector)
    private static native void predict_collect_0(long nativeObj, long src_nativeObj, long collector_nativeObj);

    // C++:  void predict(Mat src, int& label, double& confidence)
    private static native void predict_0(long nativeObj, long src_nativeObj, double[] label_out, double[] confidence_out);

    // C++:  void read(String filename)
    private static native void read_0(long nativeObj, String filename);

    // C++:  void setLabelInfo(int label, String strInfo)
    private static native void setLabelInfo_0(long nativeObj, int label, String strInfo);

    // C++:  void train(vector_Mat src, Mat labels)
    private static native void train_0(long nativeObj, long src_mat_nativeObj, long labels_nativeObj);

    // C++:  void update(vector_Mat src, Mat labels)
    private static native void update_0(long nativeObj, long src_mat_nativeObj, long labels_nativeObj);

    // C++:  void write(String filename)
    private static native void write_0(long nativeObj, String filename);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
