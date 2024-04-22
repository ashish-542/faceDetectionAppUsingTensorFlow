const express = require("express");
const { join } = require("path");
const serveStatic = require("serve-static");
const { createCanvas, loadImage } = require("canvas");
const canvas=require("canvas");
const faceapi = require("face-api.js"); 
const fetch = require('node-fetch');
const app = express();
const { JSDOM } = require("jsdom");
const PORT = 3000;
require('@tensorflow/tfjs-node');

// Serve static files from the "public" directory
app.use(express.static("public"));

// Serve the models directory
app.use("/models", serveStatic(join(__dirname, "public", "models")));

// Initialize JSDOM environment
const jsdom = new JSDOM(`<!DOCTYPE html><img id="myImg" src="photo.jpeg" />`, {
    resources: "usable",
    url: __dirname + "/"
});

global.window = jsdom.window;
global.document = window.document;

// Set up face-api.js environment
faceapi.env.monkeyPatch({
    fetch: fetch,
    Canvas: window.HTMLCanvasElement,
    Image: window.HTMLImageElement,
    ImageData: canvas.ImageData,
    createCanvasElement: () => document.createElement('canvas'),
    createImageElement: () => document.createElement('img')
});


function loadMtcnnModel() {
    // Load models asynchronously when the server starts
    console.log("Loading models from file system");
    const modelsPath = join(__dirname, "public", "models");
    Promise.all([
        faceapi.nets.ssdMobilenetv1.loadFromDisk(`${modelsPath}/ssd_mobilenetv1_model-weights_manifest.json`),
        faceapi.nets.faceLandmark68Net.loadFromDisk(`${modelsPath}/face_landmark_68_model-weights_manifest.json`),
        faceapi.nets.faceRecognitionNet.loadFromDisk(`${modelsPath}/face_recognition_model-weights_manifest.json`),
    ]).then(startServer);
}

function startServer() {
    app.listen(PORT, () => {
        console.log(`Server is running on port ${PORT}`);
    });
}

// Route to detect faces in an image
app.get("/", async (req, res) => {
    try {
        console.log("Hello");
        image = document.getElementById('myImg');
        console.log("🚀 ~ app.get ~ image:", image)
        let fullFaceDescriptions = await faceapi.detectSingleFace(image).withFaceLandmarks().withFaceDescriptor();

        res.json(fullFaceDescriptions);
    } catch (error) {
        console.log("🚀 ~ error:", error)
    }
})

loadMtcnnModel();
