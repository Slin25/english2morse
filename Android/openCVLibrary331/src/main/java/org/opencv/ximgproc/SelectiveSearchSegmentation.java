
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.ximgproc;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Algorithm;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.utils.Converters;

// C++: class SelectiveSearchSegmentation
//javadoc: SelectiveSearchSegmentation

public class SelectiveSearchSegmentation extends Algorithm {

    protected SelectiveSearchSegmentation(long addr) { super(addr); }


    //
    // C++:  void addGraphSegmentation(Ptr_GraphSegmentation g)
    //

    //javadoc: SelectiveSearchSegmentation::addGraphSegmentation(g)
    public  void addGraphSegmentation(GraphSegmentation g)
    {
        
        addGraphSegmentation_0(nativeObj, g.getNativeObjAddr());
        
        return;
    }


    //
    // C++:  void addImage(Mat img)
    //

    //javadoc: SelectiveSearchSegmentation::addImage(img)
    public  void addImage(Mat img)
    {
        
        addImage_0(nativeObj, img.nativeObj);
        
        return;
    }


    //
    // C++:  void addStrategy(Ptr_SelectiveSearchSegmentationStrategy s)
    //

    //javadoc: SelectiveSearchSegmentation::addStrategy(s)
    public  void addStrategy(SelectiveSearchSegmentationStrategy s)
    {
        
        addStrategy_0(nativeObj, s.getNativeObjAddr());
        
        return;
    }


    //
    // C++:  void clearGraphSegmentations()
    //

    //javadoc: SelectiveSearchSegmentation::clearGraphSegmentations()
    public  void clearGraphSegmentations()
    {
        
        clearGraphSegmentations_0(nativeObj);
        
        return;
    }


    //
    // C++:  void clearImages()
    //

    //javadoc: SelectiveSearchSegmentation::clearImages()
    public  void clearImages()
    {
        
        clearImages_0(nativeObj);
        
        return;
    }


    //
    // C++:  void clearStrategies()
    //

    //javadoc: SelectiveSearchSegmentation::clearStrategies()
    public  void clearStrategies()
    {
        
        clearStrategies_0(nativeObj);
        
        return;
    }


    //
    // C++:  void process(vector_Rect& rects)
    //

    //javadoc: SelectiveSearchSegmentation::process(rects)
    public  void process(MatOfRect rects)
    {
        Mat rects_mat = rects;
        process_0(nativeObj, rects_mat.nativeObj);
        
        return;
    }


    //
    // C++:  void setBaseImage(Mat img)
    //

    //javadoc: SelectiveSearchSegmentation::setBaseImage(img)
    public  void setBaseImage(Mat img)
    {
        
        setBaseImage_0(nativeObj, img.nativeObj);
        
        return;
    }


    //
    // C++:  void switchToSelectiveSearchFast(int base_k = 150, int inc_k = 150, float sigma = 0.8f)
    //

    //javadoc: SelectiveSearchSegmentation::switchToSelectiveSearchFast(base_k, inc_k, sigma)
    public  void switchToSelectiveSearchFast(int base_k, int inc_k, float sigma)
    {
        
        switchToSelectiveSearchFast_0(nativeObj, base_k, inc_k, sigma);
        
        return;
    }

    //javadoc: SelectiveSearchSegmentation::switchToSelectiveSearchFast()
    public  void switchToSelectiveSearchFast()
    {
        
        switchToSelectiveSearchFast_1(nativeObj);
        
        return;
    }


    //
    // C++:  void switchToSelectiveSearchQuality(int base_k = 150, int inc_k = 150, float sigma = 0.8f)
    //

    //javadoc: SelectiveSearchSegmentation::switchToSelectiveSearchQuality(base_k, inc_k, sigma)
    public  void switchToSelectiveSearchQuality(int base_k, int inc_k, float sigma)
    {
        
        switchToSelectiveSearchQuality_0(nativeObj, base_k, inc_k, sigma);
        
        return;
    }

    //javadoc: SelectiveSearchSegmentation::switchToSelectiveSearchQuality()
    public  void switchToSelectiveSearchQuality()
    {
        
        switchToSelectiveSearchQuality_1(nativeObj);
        
        return;
    }


    //
    // C++:  void switchToSingleStrategy(int k = 200, float sigma = 0.8f)
    //

    //javadoc: SelectiveSearchSegmentation::switchToSingleStrategy(k, sigma)
    public  void switchToSingleStrategy(int k, float sigma)
    {
        
        switchToSingleStrategy_0(nativeObj, k, sigma);
        
        return;
    }

    //javadoc: SelectiveSearchSegmentation::switchToSingleStrategy()
    public  void switchToSingleStrategy()
    {
        
        switchToSingleStrategy_1(nativeObj);
        
        return;
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  void addGraphSegmentation(Ptr_GraphSegmentation g)
    private static native void addGraphSegmentation_0(long nativeObj, long g_nativeObj);

    // C++:  void addImage(Mat img)
    private static native void addImage_0(long nativeObj, long img_nativeObj);

    // C++:  void addStrategy(Ptr_SelectiveSearchSegmentationStrategy s)
    private static native void addStrategy_0(long nativeObj, long s_nativeObj);

    // C++:  void clearGraphSegmentations()
    private static native void clearGraphSegmentations_0(long nativeObj);

    // C++:  void clearImages()
    private static native void clearImages_0(long nativeObj);

    // C++:  void clearStrategies()
    private static native void clearStrategies_0(long nativeObj);

    // C++:  void process(vector_Rect& rects)
    private static native void process_0(long nativeObj, long rects_mat_nativeObj);

    // C++:  void setBaseImage(Mat img)
    private static native void setBaseImage_0(long nativeObj, long img_nativeObj);

    // C++:  void switchToSelectiveSearchFast(int base_k = 150, int inc_k = 150, float sigma = 0.8f)
    private static native void switchToSelectiveSearchFast_0(long nativeObj, int base_k, int inc_k, float sigma);
    private static native void switchToSelectiveSearchFast_1(long nativeObj);

    // C++:  void switchToSelectiveSearchQuality(int base_k = 150, int inc_k = 150, float sigma = 0.8f)
    private static native void switchToSelectiveSearchQuality_0(long nativeObj, int base_k, int inc_k, float sigma);
    private static native void switchToSelectiveSearchQuality_1(long nativeObj);

    // C++:  void switchToSingleStrategy(int k = 200, float sigma = 0.8f)
    private static native void switchToSingleStrategy_0(long nativeObj, int k, float sigma);
    private static native void switchToSingleStrategy_1(long nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
