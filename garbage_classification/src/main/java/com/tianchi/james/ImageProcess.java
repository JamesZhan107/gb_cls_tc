package com.tianchi.james;

import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat;
import com.intel.analytics.zoo.feature.image.OpenCVMethod;
import com.intel.analytics.zoo.pipeline.inference.JTensor;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.Raster;
import java.awt.image.WritableRaster;
import java.io.IOException;

//OpenCV读取 + BufferedImage获取像素
// boolean ifReverseInputChannels =false
public class ImageProcess {
    final static int width = 299;
    final static int height = 299;

    public static JTensor process(byte[] imageBytes) throws IOException {
        //图片读取
        long time1 = System.currentTimeMillis();
        OpenCVMat openCVMat = OpenCVMethod.fromImageBytes(imageBytes, Imgcodecs.CV_LOAD_IMAGE_UNCHANGED);
        long time2 = System.currentTimeMillis();

        //剪裁
        Mat dst = new Mat();
        Imgproc.resize(openCVMat, dst, new Size(299,299));
        long time3 = System.currentTimeMillis();

        BufferedImage image = matToBufferedImage(dst);
        Raster raster = image.getData();

        float[] ftmp = new float[raster.getWidth() * raster.getHeight() * raster.getNumBands()];
        float[] fpixels = raster.getPixels(0, 0, raster.getWidth(), raster.getHeight(), ftmp);
        float[] ftmpCHW = fromHWC2CHW(fpixels);
        JTensor tensor = new JTensor(ftmpCHW, new int[]{1, width, height, 3});
        long time4 = System.currentTimeMillis();

        System.out.println("读取图片........" + (time2-time1));
        System.out.println("剪裁图片........" + (time3-time2));
        System.out.println("获取像素值........" + (time4-time3));
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

    //Mat转换为BufferedImage
    public static BufferedImage matToBufferedImage(Mat mat)
    {
        if (mat.height() > 0 && mat.width() > 0)
        {
            BufferedImage image = new BufferedImage(mat.width(), mat.height(), BufferedImage.TYPE_3BYTE_BGR);
            WritableRaster raster = image.getRaster();
            DataBufferByte dataBuffer = (DataBufferByte) raster.getDataBuffer();
            byte[] data = dataBuffer.getData();
            mat.get(0, 0, data);
            return image;
        }
        return null;
    }
}
