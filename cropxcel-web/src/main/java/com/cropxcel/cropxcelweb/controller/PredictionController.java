package com.cropxcel.cropxcelweb.controller;

import org.springframework.stereotype.Controller;
import java.awt.image.BufferedImage;
import java.io.BufferedOutputStream;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Base64;
import java.util.List;
import java.util.UUID;

import javax.imageio.ImageIO;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.framework.ConfigProto;
import org.tensorflow.framework.GPUOptions;

import com.cropxcel.cropxcelweb.aiml.Prediction;
import com.google.gson.Gson;

@Controller
public class PredictionController {
    private static final String MODEL_FILE = "model.h5";
    private static final int IMAGE_SIZE = 224;
    private static final int CHANNELS = 3;

    @Autowired
    public Prediction prediction;

    @PostMapping(value = "/predict")
    public ResponseEntity<Prediction> predict(@RequestBody String base64Image) {

        try {
            String imageFileName = UUID.randomUUID().toString() + ".jpg";
            String imagePath = "images/" + imageFileName;
            ImageUtils.saveImage(base64Image, imagePath);

            Graph graph = new Graph();
            byte[] graphBytes = Files.readAllBytes(Paths.get(MODEL_FILE));
            graph.importGraphDef(graphBytes);

            try (Session session = new Session(graph, new Session.SessionOptions())) {
                List<Tensor<?>> outputs = session.runner()
                        .feed("input_1", normalizedImage)
                        .fetch("dense_2/BiasAdd")
                        .run();

                Tensor<Float> predictions = outputs.get(0).expect(Float.class);
                float[][] predictionValues = new float[1][(int) predictions.shape()[1]];
                predictions.copyTo(predictionValues);

                String[] classLabels = { "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___healthy",
                        "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
                        "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
                        "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus" };

                Prediction prediction = new Prediction(classLabels, predictionValues[0]);

                return new ResponseEntity<Prediction>(prediction, HttpStatus.OK);
            }

        } catch (IOException e) {
            e.printStackTrace();
            return new ResponseEntity<Prediction>(HttpStatus.INTERNAL_SERVER_ERROR);
        }

    }

    public static Tensor readNormalizeImage(String imagePath, int imageSize, int channels) throws IOException {
        float[][][][] image = new float[1][imageSize][imageSize][channels];
        readImage(imagePath, image);
        return Tensor.create(image).div(255.0f);
    }

    private static class ImageUtils {

        public static void saveImage(String base64Image, String path) throws IOException {
            byte[] imageBytes = Base64.getDecoder().decode(base64Image.split(",")[1]);
            try (BufferedOutputStream stream = new BufferedOutputStream(new FileOutputStream(path))) {
                stream.write(imageBytes);
            }
        }

        public static Tensor<Float> readNormalizeImage(String imagePath, int imageSize, int channels)
                throws IOException {
            float[][][][] image = new float[1][imageSize][imageSize][channels];
            ImageUtils.readImage(imagePath, image);
            return Tensor.create(image).div(255.0f);
        }

        private static void readImage(String imagePath, float[][][][] image) throws IOException {
            int[] values = new int[3];
            File file = new File(imagePath);
            BufferedImage bufferedImage = ImageIO.read(file);
            int width = bufferedImage.getWidth();
            int height = bufferedImage.getHeight();

            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    int pixel = bufferedImage.getRGB(x, y);
                    values[0] = (pixel >> 16) & 0xFF; // Red component
                    values[1] = (pixel >> 8) & 0xFF; // Green component
                    values[2] = pixel & 0xFF; // Blue component
                    for (int c = 0; c < 3; c++) {
                        image[0][y][x][c] = (values[c] - IMAGE_MEAN[c]) / IMAGE_STD[c];
                    }
                }
            }
        }
    }
}