
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.face;



// C++: class StandardCollector
//javadoc: StandardCollector

public class StandardCollector extends PredictCollector {

    protected StandardCollector(long addr) { super(addr); }


    //
    // C++: static Ptr_StandardCollector create(double threshold = DBL_MAX)
    //

    //javadoc: StandardCollector::create(threshold)
    public static StandardCollector create(double threshold)
    {
        
        StandardCollector retVal = new StandardCollector(create_0(threshold));
        
        return retVal;
    }

    //javadoc: StandardCollector::create()
    public static StandardCollector create()
    {
        
        StandardCollector retVal = new StandardCollector(create_1());
        
        return retVal;
    }


    //
    // C++:  double getMinDist()
    //

    //javadoc: StandardCollector::getMinDist()
    public  double getMinDist()
    {
        
        double retVal = getMinDist_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  int getMinLabel()
    //

    //javadoc: StandardCollector::getMinLabel()
    public  int getMinLabel()
    {
        
        int retVal = getMinLabel_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  vector_pair_int_and_double getResults(bool sorted = false)
    //

    // Return type 'vector_pair_int_and_double' is not supported, skipping the function


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++: static Ptr_StandardCollector create(double threshold = DBL_MAX)
    private static native long create_0(double threshold);
    private static native long create_1();

    // C++:  double getMinDist()
    private static native double getMinDist_0(long nativeObj);

    // C++:  int getMinLabel()
    private static native int getMinLabel_0(long nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
