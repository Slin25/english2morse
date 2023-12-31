
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.tracking;



// C++: class TrackerKCF
//javadoc: TrackerKCF

public class TrackerKCF extends Tracker {

    protected TrackerKCF(long addr) { super(addr); }


    public static final int
            GRAY = (1 << 0),
            CN = (1 << 1),
            CUSTOM = (1 << 2);


    //
    // C++: static Ptr_TrackerKCF create()
    //

    //javadoc: TrackerKCF::create()
    public static TrackerKCF create()
    {
        
        TrackerKCF retVal = new TrackerKCF(create_0());
        
        return retVal;
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++: static Ptr_TrackerKCF create()
    private static native long create_0();

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
