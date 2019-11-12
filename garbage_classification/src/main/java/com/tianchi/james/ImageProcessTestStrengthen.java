package com.tianchi.james;

import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat;
import com.intel.analytics.zoo.feature.image.ImageSet;
import com.intel.analytics.zoo.feature.image.OpenCVMethod;
import com.intel.analytics.zoo.pipeline.inference.JTensor;
import org.imgscalr.Scalr;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import scala.Array;
import scala.collection.immutable.HashMap;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.Raster;
import java.awt.image.WritableRaster;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.ArrayList;

public class ImageProcessTestStrengthen {
    final static int width = 359;
    final static int height = 359;

    public static ArrayList<JTensor> process(byte[] imageBytes) throws IOException {
        //图片读取方法1
//        long time1 = System.currentTimeMillis();
//        ByteArrayInputStream in = new ByteArrayInputStream(imageBytes);
//        BufferedImage image = ImageIO.read(in);
//        long time2 = System.currentTimeMillis();

        //图片读取方法2
        //long time1 = System.currentTimeMillis();
        OpenCVMat openCVMat = OpenCVMethod.fromImageBytes(imageBytes, Imgcodecs.CV_LOAD_IMAGE_UNCHANGED);
        //long time2 = System.currentTimeMillis();

        //剪裁法1
        Mat dst=new Mat();
        Imgproc.resize(openCVMat, dst, new Size(width,height));

        //剪裁法2
//        BufferedImage image = matToBufferedImage(openCVMat);
//        BufferedImage thumbnail = Scalr.resize(image, Scalr.Method.SPEED, Scalr.Mode.FIT_EXACT, width, height);
//        Raster raster = thumbnail.getData();

        //获取像素值
        float[] fpixels = new float[width * height * 3];
        OpenCVMat.toFloatPixels(dst, fpixels);
        float[] ftmpCHW = fromHWC2CHW(fpixels);
        JTensor tensor = new JTensor(ftmpCHW, new int[]{1, width, height, 3});
        //long time3 = System.currentTimeMillis();

        //翻转
        Mat dst_flip=new Mat();
        Core.flip(dst, dst_flip, 1);

        //获取像素值
        float[] fpixels_flip = new float[width * height * 3];
        OpenCVMat.toFloatPixels(dst_flip, fpixels_flip);
        float[] ftmpCHW_flip = fromHWC2CHW(fpixels_flip);
        JTensor tensor_flip = new JTensor(ftmpCHW_flip, new int[]{1, width, height, 3});
        //long time4 = System.currentTimeMillis();

        ArrayList<JTensor> arrayList = new ArrayList<>();
        arrayList.add(tensor);
        arrayList.add(tensor_flip);
//        System.out.println("读取图片........" + (time2-time1));
//        System.out.println("剪裁图片........" + (time3-time2));
//        System.out.println("翻转图片........" + (time4-time3));
        return arrayList;
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
