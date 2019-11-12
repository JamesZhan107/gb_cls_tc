package com.tianchi.james;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;

public class Util {
    public static int indexOffMax(float[] outputDatas) {
        int maxAt = 0;
        for(int i =0; i < outputDatas.length; i++ ){
            maxAt = outputDatas[i] > outputDatas[maxAt] ? i : maxAt;
        }
        return maxAt;
    }

    public static Map<Integer, String> loadClassDict() throws IOException {
        Map<Integer, String> map = new HashMap();
        InputStream fileInputStream = ZooMain.class.getClassLoader().getResourceAsStream("class_index.txt");
        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(fileInputStream));
        String str = null;
        while((str = bufferedReader.readLine()) != null){
            String[] tmp = str.split(" ");
            map.put(Integer.valueOf(tmp[1]), tmp[0]);
        }
        fileInputStream.close();
        bufferedReader.close();
        return map;
    }
}
