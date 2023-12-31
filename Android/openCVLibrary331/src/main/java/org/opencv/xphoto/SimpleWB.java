
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.xphoto;



// C++: class SimpleWB
//javadoc: SimpleWB

public class SimpleWB extends WhiteBalancer {

    protected SimpleWB(long addr) { super(addr); }


    //
    // C++:  float getInputMax()
    //

    //javadoc: SimpleWB::getInputMax()
    public  float getInputMax()
    {
        
        float retVal = getInputMax_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  float getInputMin()
    //

    //javadoc: SimpleWB::getInputMin()
    public  float getInputMin()
    {
        
        float retVal = getInputMin_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  float getOutputMax()
    //

    //javadoc: SimpleWB::getOutputMax()
    public  float getOutputMax()
    {
        
        float retVal = getOutputMax_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  float getOutputMin()
    //

    //javadoc: SimpleWB::getOutputMin()
    public  float getOutputMin()
    {
        
        float retVal = getOutputMin_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  float getP()
    //

    //javadoc: SimpleWB::getP()
    public  float getP()
    {
        
        float retVal = getP_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  void setInputMax(float val)
    //

    //javadoc: SimpleWB::setInputMax(val)
    public  void setInputMax(float val)
    {
        
        setInputMax_0(nativeObj, val);
        
        return;
    }


    //
    // C++:  void setInputMin(float val)
    //

    //javadoc: SimpleWB::setInputMin(val)
    public  void setInputMin(float val)
    {
        
        setInputMin_0(nativeObj, val);
        
        return;
    }


    //
    // C++:  void setOutputMax(float val)
    //

    //javadoc: SimpleWB::setOutputMax(val)
    public  void setOutputMax(float val)
    {
        
        setOutputMax_0(nativeObj, val);
        
        return;
    }


    //
    // C++:  void setOutputMin(float val)
    //

    //javadoc: SimpleWB::setOutputMin(val)
    public  void setOutputMin(float val)
    {
        
        setOutputMin_0(nativeObj, val);
        
        return;
    }


    //
    // C++:  void setP(float val)
    //

    //javadoc: SimpleWB::setP(val)
    public  void setP(float val)
    {
        
        setP_0(nativeObj, val);
        
        return;
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  float getInputMax()
    private static native float getInputMax_0(long nativeObj);

    // C++:  float getInputMin()
    private static native float getInputMin_0(long nativeObj);

    // C++:  float getOutputMax()
    private static native float getOutputMax_0(long nativeObj);

    // C++:  float getOutputMin()
    private static native float getOutputMin_0(long nativeObj);

    // C++:  float getP()
    private static native float getP_0(long nativeObj);

    // C++:  void setInputMax(float val)
    private static native void setInputMax_0(long nativeObj, float val);

    // C++:  void setInputMin(float val)
    private static native void setInputMin_0(long nativeObj, float val);

    // C++:  void setOutputMax(float val)
    private static native void setOutputMax_0(long nativeObj, float val);

    // C++:  void setOutputMin(float val)
    private static native void setOutputMin_0(long nativeObj, float val);

    // C++:  void setP(float val)
    private static native void setP_0(long nativeObj, float val);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
