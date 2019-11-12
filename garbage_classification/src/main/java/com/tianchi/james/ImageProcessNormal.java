package com.tianchi.james;

import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat;
import com.intel.analytics.zoo.feature.image.OpenCVMethod;
import com.intel.analytics.zoo.pipeline.inference.JTensor;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import java.io.IOException;

//OpenCV读取 + OpenCV获取像素
// boolean ifReverseInputChannels =true
public class ImageProcessNormal {
    final static int width = 369;
    final static int height = 369;

    public static JTensor process(byte[] imageBytes) throws IOException {
        //图片读取
        //long time1 = System.currentTimeMillis();
        OpenCVMat openCVMat = OpenCVMethod.fromImageBytes(imageBytes, Imgcodecs.CV_LOAD_IMAGE_UNCHANGED);
        //long time2 = System.currentTimeMillis();

        //剪裁
        Mat dst = new Mat();
        Imgproc.resize(openCVMat, dst, new Size(width,height));
        //long time3 = System.currentTimeMillis();

        //获取像素值
        float[] fpixels = new float[width * height * 3];
        OpenCVMat.toFloatPixels(dst, fpixels);
        float[] ftmpCHW = fromHWC2CHW(fpixels);
        JTensor tensor = new JTensor(ftmpCHW, new int[]{1, width, height, 3});
        //long time5 = System.currentTimeMillis();

//        System.out.println("读取图片........" + (time2-time1));
//        System.out.println("剪裁图片........" + (time3-time2));
//        System.out.println("获取像素值........" + (time5-time3));
        return tensor;
    }

    public static float[] fromHWC2CHW(float[] data) {
        float[] resArray = new float[3 * width * height];
        for (int h = 0; h <= height-1; h++) {
            for (int w = 0; w <= width-1; w++) {
                for (int c = 0; c <= 2; c++) {
                    resArray[c * height * width + h * width + w] = (data[h * width * 3 + w * 3 + c]/127.5f) - 1.0f;
                }
            }
        }
        return resArray;
    }
}