package com.synthmodeler.synth_modeler_api.service;

import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.stereotype.Service;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.types.TFloat32;
import java.nio.FloatBuffer;
import org.springframework.beans.factory.annotation.Autowired;

@Service
public class ModelService {
    private final SavedModelBundle model;
    private final Session session;

    @Autowired
    public ModelService(ResourceLoader resourceLoader) {
        try {
            Resource resource = resourceLoader.getResource("classpath:models/saved_model_20250601-181033"); // Replace with actual timestamp
            if (!resource.exists()) {
                throw new IllegalStateException("SavedModel not found at: models/saved_model_20250601-181033");
            }
            String modelPath = resource.getFile().getAbsolutePath();
            model = SavedModelBundle.load(modelPath, "serve");
            session = model.session();
        } catch (Exception e) {
            throw new IllegalStateException("Failed to load SavedModel: " + e.getMessage(), e);
        }
    }

    public float[][][] predict(float[][][] mel, float[] f0, float[] loudness) {
        int batchSize = mel.length;
        int timeSteps = mel[0].length;
        int melFeatures = 128;

        // Validate inputs
        if (mel[0][0].length != melFeatures) {
            throw new IllegalArgumentException("Mel spectrogram must have 128 features");
        }
        if (f0.length != batchSize * timeSteps || loudness.length != batchSize * timeSteps) {
            throw new IllegalArgumentException("F0 and loudness must match batchSize * timeSteps");
        }

        // Prepare inputs
        FloatBuffer melInput = FloatBuffer.allocate(batchSize * timeSteps * melFeatures);
        FloatBuffer f0Input = FloatBuffer.allocate(batchSize * timeSteps);
        FloatBuffer loudnessInput = FloatBuffer.allocate(batchSize * timeSteps);

        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < timeSteps; t++) {
                for (int f = 0; f < melFeatures; f++) {
                    melInput.put(mel[b][t][f]);
                }
                f0Input.put(f0[b * timeSteps + t]);
                loudnessInput.put(loudness[b * timeSteps + t]);
            }
        }
        melInput.rewind();
        f0Input.rewind();
        loudnessInput.rewind();

        // Create input tensors
        try (Tensor melTensor = TFloat32.create(new long[]{batchSize, timeSteps, melFeatures}, melInput);
             Tensor f0Tensor = TFloat32.create(new long[]{batchSize, timeSteps}, f0Input);
             Tensor loudnessTensor = TFloat32.create(new long[]{batchSize, timeSteps}, loudnessInput)) {

            // Run inference
            int oscSize = 10; // Adjust per config.toml
            int filterSize = 5;
            int fxSize = 3;
            float[][] oscOutput = new float[batchSize][oscSize];
            float[][] filterOutput = new float[batchSize][filterSize];
            float[][] fxOutput = new float[batchSize][fxSize];

            var runner = session.runner()
                    .feed("mel_input", melTensor)
                    .feed("f0_input", f0Tensor)
                    .feed("loudness_input", loudnessTensor)
                    .fetch("osc_output")
                    .fetch("filter_output")
                    .fetch("fx_output");

            var results = runner.run();
            try (Tensor oscResult = results.get(0);
                 Tensor filterResult = results.get(1);
                 Tensor fxResult = results.get(2)) {
                oscResult.copyTo(oscOutput);
                filterResult.copyTo(filterOutput);
                fxResult.copyTo(fxOutput);
            }

            return new float[][][]{oscOutput, filterOutput, fxOutput};
        }
    }
}