package com.cropxcel.cropxcelweb.aiml;

public class Prediction {
    private String[] classLabels;
    private float[] predictionValues;

    public Prediction(String[] classLabels, float[] predictionValues) {
        this.classLabels = classLabels;
        this.predictionValues = predictionValues;
    }

    public String[] getClassLabels() {
        return classLabels;
    }

    public void setClassLabels(String[] classLabels) {
        this.classLabels = classLabels;
    }

    public float[] getPredictionValues() {
        return predictionValues;
    }

    public void setPredictionValues(float[] predictionValues) {
        this.predictionValues = predictionValues;
    }
}
