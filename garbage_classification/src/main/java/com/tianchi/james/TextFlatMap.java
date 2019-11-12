package com.tianchi.james;

import com.alibaba.tianchi.garbage_image_util.IdLabel;
import com.alibaba.tianchi.garbage_image_util.ImageData;
import com.intel.analytics.zoo.pipeline.inference.JTensor;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.util.Collector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class TextFlatMap extends RichFlatMapFunction<ImageData, IdLabel> {
    private Map<Integer, String> indexClassDict;
    private byte[] saveModelTarBytes;
    private String saveModelPath;
    private String input;
    private int[] inputShape;
    private boolean ifReverseInputChannels;
    private float[] meanValues;
    private float scale;
    private GarbageClassificationInferenceModel model;


    public TextFlatMap(byte[] saveModelTarBytes, int[] inputShape, boolean ifReverseInputChannels, float[] meanValues, float scale, String input) {
        this.saveModelTarBytes = saveModelTarBytes;
        this.input = input;
        this.inputShape = inputShape;
        this.ifReverseInputChannels = ifReverseInputChannels;
        this.meanValues = meanValues;
        this.scale = scale;
    }

    public TextFlatMap(String input, int[] inputShape, boolean ifReverseInputChannels, float[] meanValues, float scale) {
        this.saveModelTarBytes = null;
        this.saveModelPath = System.getenv("MODEL_INFERENCE_PATH") + "/SavedModel";
        if(null == saveModelPath){
            throw new RuntimeException("No IMAGE_MODEL_PATH");
        }else {
            System.out.println(String.format("IMAGE_MODEL_PATH %s", saveModelPath));
        }
        this.input = input;
        this.inputShape = inputShape;
        this.ifReverseInputChannels = ifReverseInputChannels;
        this.meanValues = meanValues;
        this.scale = scale;
    }

    //测试增强
    @Override
    public void flatMap(ImageData value, Collector<IdLabel> out) throws Exception {
        //long time1 = System.currentTimeMillis();
        ArrayList<JTensor> imageTensorList = ImageProcessTestStrengthen.process(value.getImage());
        //long time2 = System.currentTimeMillis();

        List<JTensor> data = Arrays.asList(imageTensorList.get(0));
        List<List<JTensor>> inputs = new ArrayList<>();
        inputs.add(data);

        List<JTensor> data_flip = Arrays.asList(imageTensorList.get(1));
        List<List<JTensor>> inputs_flip = new ArrayList<>();
        inputs_flip.add(data_flip);

        float[] outputDatas = model.predict(inputs).get(0).get(0).getData();
        float[] outputDatas_flip = model.predict(inputs_flip).get(0).get(0).getData();
        //long time3 = System.currentTimeMillis();

        float[] output = sumAndAverage(outputDatas, outputDatas_flip);
        int index = Util.indexOffMax(output);
        String label = indexClassDict.get(index);

        IdLabel idLabel = new IdLabel((value.getId()), label);
        out.collect(idLabel);
        //long time4 = System.currentTimeMillis();
//        System.out.println("图片处理时间---------------" + (time2-time1));
//        System.out.println("图片预测时间---------------" + (time3-time2));
//        System.out.println("总耗时---------------" + (time4-time1) + "\\n");
    }

    private float[] sumAndAverage(float[] outputDatas, float[] outputDatas_flip) {
        float[] outputArray = new float[100];
        for (int i = 0; i < outputDatas.length; i++){
            float data = outputDatas[i];
            float data_flip = outputDatas_flip[i];
            float out = (data + data_flip) / 2.0f;
            outputArray[i] = out;
        }
        return outputArray;
    }

//    @Override
//    public void flatMap(ImageData value, Collector<IdLabel> out) throws Exception {
//        //long time1 = System.currentTimeMillis();
//        JTensor imageTensor = ImageProcessNormal.process(value.getImage());
//        //long time2 = System.currentTimeMillis();
//
//        List<JTensor> data = Arrays.asList(imageTensor);
//        List<List<JTensor>> inputs = new ArrayList<>();
//        inputs.add(data);
//
//        float[] outputDatas = model.predict(inputs).get(0).get(0).getData();
//        //long time3 = System.currentTimeMillis();
//
//        int index = Util.indexOffMax(outputDatas);
//        String label = indexClassDict.get(index);
//
//        IdLabel idLabel = new IdLabel((value.getId()), label);
//        out.collect(idLabel);
//        //long time4 = System.currentTimeMillis();
////        System.out.println("图片处理时间---------------" + (time2-time1));
////        System.out.println("图片预测时间---------------" + (time3-time2));
////        System.out.println("总耗时---------------" + (time4-time1));
//    }

    @Override
    public void open(Configuration parameters) throws Exception{
        indexClassDict = Util.loadClassDict();
        model = new GarbageClassificationInferenceModel();
        if(null == saveModelTarBytes){
            System.out.println("doLoadTF_1");
            //model.doLoadTF(saveModelPath);
            model.doLoadTF(saveModelPath, inputShape, ifReverseInputChannels, meanValues, scale, input);
        }else {
            model.doLoadTF(saveModelTarBytes, inputShape, ifReverseInputChannels, meanValues, scale, input);
        }
    }

    @Override
    public void close() throws Exception{
        model.release();
    }
}
