package com.tianchi.james;

import com.intel.analytics.zoo.pipeline.inference.InferenceModel;
import com.intel.analytics.zoo.pipeline.inference.JTensor;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class GarbageClassificationInferenceModel extends InferenceModel implements Serializable {
    public GarbageClassificationInferenceModel() {
        super();
    }

    public GarbageClassificationInferenceModel(int concurrentNum) {
        super(concurrentNum);
    }

    public void release() {
        doRelease();
    }

    @Deprecated
    public List<Float> predict(List<Float> input, int... shape) {
        List<Integer> inputShape = new ArrayList<Integer>();
        for (int s : shape) {
            inputShape.add(s);
        }
        return doPredict(input, inputShape);
    }

    public List<List<JTensor>> predict(List<List<JTensor>> inputs) {
        return doPredict(inputs);
    }

    public List<List<JTensor>> predict(List<JTensor>[] inputs) {
        return predict(Arrays.asList(inputs));
    }

    @Override
    public String toString() {
        return super.toString();
    }
}
