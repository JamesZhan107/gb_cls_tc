import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat;
import com.intel.analytics.zoo.feature.image.OpenCVMethod;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.Raster;
import java.awt.image.WritableRaster;
import java.io.ByteArrayOutputStream;
import java.io.File;
//import org.opencv.highgui.HighGui;


public class zooTest {
    static byte[] bytes;

    public static void main(String[] args) throws Exception {
        File img = new File("E:\\tianchi\\2.jpg");
        byte[] imageBytes = fileToByte(img);
        OpenCVMat openCVMat = OpenCVMethod.fromImageBytes(imageBytes, Imgcodecs.CV_LOAD_IMAGE_UNCHANGED);

        //剪裁
        Mat dst = new Mat();
        Imgproc.resize(openCVMat, dst, new Size(299,299));

        Mat dst1 = new Mat();
        Imgproc.cvtColor(dst,dst1,Imgproc.COLOR_BGR2RGB);
        //获取像素值1
//        float[] fpixels = new float[299 * 299 * 3];
//        OpenCVMat.toFloatPixels(dst1, fpixels);
//        System.out.println("test1...." + fpixels.length);
//        for (int i = 0; i < fpixels.length; i++) {
//            System.out.print(fpixels[i] + ",");
//        }

        //获取像素值2
        BufferedImage image = matToBufferedImage(dst);
        Raster raster = image.getData();
        float[] ftmp = new float[raster.getWidth() * raster.getHeight() * raster.getNumBands()];
        float[] fpixels2 = raster.getPixels(0, 0, raster.getWidth(), raster.getHeight(), ftmp);
        System.out.println("test2...." + fpixels2.length);
        for (int i = 0; i < fpixels2.length; i++) {
            System.out.print(fpixels2[i] + ",");
        }

//        ImageProcessingTest imageProcessingTest = new ImageProcessingTest();
//        long time1 = System.currentTimeMillis();
//        OpenCVMat openCVMat = imageProcessingTest.byteArrayToMat(imageBytes, Imgcodecs.CV_LOAD_IMAGE_UNCHANGED);
//        long time2 = System.currentTimeMillis();
//        Mat dst=openCVMat.clone();
//        Imgproc.resize(openCVMat, dst, new Size(400,400));
//        BufferedImage image = matToBufferedImage(dst);


//        // resie 224*224
//        BufferedImage thumbnail = Scalr.resize(image, Scalr.Method.SPEED, Scalr.Mode.FIT_EXACT, 299, 299);
        //ImageIO.write(image, "jpg", new File("E:\\tianchi\\22.jpg"));
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

    public static BufferedImage mat2BI(Mat mat){
        int dataSize =mat.cols()*mat.rows()*(int)mat.elemSize();
        byte[] data=new byte[dataSize];
        mat.get(0, 0,data);
        int type=mat.channels()==1?
                BufferedImage.TYPE_BYTE_GRAY:BufferedImage.TYPE_3BYTE_BGR;

        if(type==BufferedImage.TYPE_3BYTE_BGR){
            for(int i=0;i<dataSize;i+=3){
                byte blue=data[i+0];
                data[i+0]=data[i+2];
                data[i+2]=blue;
            }
        }
        BufferedImage image=new BufferedImage(mat.cols(),mat.rows(),type);
        image.getRaster().setDataElements(0, 0, mat.cols(), mat.rows(), data);

        return image;
    }

    public static byte[] fileToByte(File img) throws Exception {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try {
            BufferedImage bi;
            bi = ImageIO.read(img);
            ImageIO.write(bi, "jpg", baos);
            bytes = baos.toByteArray();
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            baos.close();
        }
        return bytes;
    }
}
