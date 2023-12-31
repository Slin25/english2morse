
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.face;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.utils.Converters;

// C++: class BasicFaceRecognizer
//javadoc: BasicFaceRecognizer

public class BasicFaceRecognizer extends FaceRecognizer {

    protected BasicFaceRecognizer(long addr) { super(addr); }


    //
    // C++:  Mat getEigenValues()
    //

    //javadoc: BasicFaceRecognizer::getEigenValues()
    public  Mat getEigenValues()
    {
        
        Mat retVal = new Mat(getEigenValues_0(nativeObj));
        
        return retVal;
    }


    //
    // C++:  Mat getEigenVectors()
    //

    //javadoc: BasicFaceRecognizer::getEigenVectors()
    public  Mat getEigenVectors()
    {
        
        Mat retVal = new Mat(getEigenVectors_0(nativeObj));
        
        return retVal;
    }


    //
    // C++:  Mat getLabels()
    //

    //javadoc: BasicFaceRecognizer::getLabels()
    public  Mat getLabels()
    {
        
        Mat retVal = new Mat(getLabels_0(nativeObj));
        
        return retVal;
    }


    //
    // C++:  Mat getMean()
    //

    //javadoc: BasicFaceRecognizer::getMean()
    public  Mat getMean()
    {
        
        Mat retVal = new Mat(getMean_0(nativeObj));
        
        return retVal;
    }


    //
    // C++:  double getThreshold()
    //

    //javadoc: BasicFaceRecognizer::getThreshold()
    public  double getThreshold()
    {
        
        double retVal = getThreshold_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  int getNumComponents()
    //

    //javadoc: BasicFaceRecognizer::getNumComponents()
    public  int getNumComponents()
    {
        
        int retVal = getNumComponents_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  vector_Mat getProjections()
    //

    //javadoc: BasicFaceRecognizer::getProjections()
    public  List<Mat> getProjections()
    {
        List<Mat> retVal = new ArrayList<Mat>();
        Mat retValMat = new Mat(getProjections_0(nativeObj));
        Converters.Mat_to_vector_Mat(retValMat, retVal);
        return retVal;
    }


    //
    // C++:  void setNumComponents(int val)
    //

    //javadoc: BasicFaceRecognizer::setNumComponents(val)
    public  void setNumComponents(int val)
    {
        
        setNumComponents_0(nativeObj, val);
        
        return;
    }


    //
    // C++:  void setThreshold(double val)
    //

    //javadoc: BasicFaceRecognizer::setThreshold(val)
    public  void setThreshold(double val)
    {
        
        setThreshold_0(nativeObj, val);
        
        return;
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  Mat getEigenValues()
    private static native long getEigenValues_0(long nativeObj);

    // C++:  Mat getEigenVectors()
    private static native long getEigenVectors_0(long nativeObj);

    // C++:  Mat getLabels()
    private static native long getLabels_0(long nativeObj);

    // C++:  Mat getMean()
    private static native long getMean_0(long nativeObj);

    // C++:  double getThreshold()
    private static native double getThreshold_0(long nativeObj);

    // C++:  int getNumComponents()
    private static native int getNumComponents_0(long nativeObj);

    // C++:  vector_Mat getProjections()
    private static native long getProjections_0(long nativeObj);

    // C++:  void setNumComponents(int val)
    private static native void setNumComponents_0(long nativeObj, int val);

    // C++:  void setThreshold(double val)
    private static native void setThreshold_0(long nativeObj, double val);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
