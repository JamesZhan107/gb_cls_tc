package com.tianchi.james;

import com.alibaba.tianchi.garbage_image_util.ConfigConstant;
import com.alibaba.tianchi.garbage_image_util.ImageClassSink;
import com.alibaba.tianchi.garbage_image_util.ImageDirSource;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;

public class ZooMain {
        public static void main(String[] args) throws Exception {
            //String saveModelTarPath = System.getenv("IMAGE_TRAIN_INPUT_PATH") + "/SavedModel";
            //System.out.println(saveModelTarPath);
            boolean ifReverseInputChannels = true;//当使用OpenCV获取像素值时，为true
            int[] inputShape = {1, 359, 359, 3};
            float[] meanValues = {0f, 0f, 0f};
            float scale = 1.0f;
            String input = "input_1";

//            long fileSize = new File(saveModelTarPath).length();
//            InputStream inputStream = new FileInputStream(saveModelTarPath);
//            byte[] saveModelTarBytes = new byte[(int)fileSize];
//            inputStream.read(saveModelTarBytes);

            StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
            env.disableOperatorChaining();
            env.setParallelism(1);
            ImageDirSource source = new ImageDirSource();
            env.addSource(source).setParallelism(1)
                    //1.图片处理成模型输入格式 2.模型预测
                    //.flatMap(new TextFlatMap(saveModelTarBytes, inputShape, ifReverseInputChannels, meanValues, scale, input))
                    .flatMap(new TextFlatMap(input, inputShape, ifReverseInputChannels, meanValues, scale))
                    .setParallelism(1)
                    .addSink(new ImageClassSink()).setParallelism(1);
            env.execute();
        }
}
