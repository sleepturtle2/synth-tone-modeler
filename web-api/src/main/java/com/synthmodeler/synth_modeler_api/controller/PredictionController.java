package com.synthmodeler.synth_modeler_api.controller;

import com.synthmodeler.synth_modeler_api.service.ModelService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class PredictionController {
    private final ModelService modelService;

    @Autowired
    public PredictionController(ModelService modelService) {
        this.modelService = modelService;
    }

    @PostMapping("/predict")
    public float[][][] predict(@RequestBody PredictionRequest request) throws Exception {
        return modelService.predict(request.getMel(), request.getF0(), request.getLoudness());
    }

    static class PredictionRequest {
        private float[][][] mel;
        private float[] f0;
        private float[] loudness;

        public float[][][] getMel() { return mel; }
        public void setMel(float[][][] mel) { this.mel = mel; }
        public float[] getF0() { return f0; }
        public void setF0(float[] f0) { this.f0 = f0; }
        public float[] getLoudness() { return loudness; }
        public void setLoudness(float[] loudness) { this.loudness = loudness; }
    }
}