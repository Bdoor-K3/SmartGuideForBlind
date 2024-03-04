import * as cocoSsd from '@tensorflow-models/coco-ssd';
import * as tf from '@tensorflow/tfjs';
import { Camera } from 'expo-camera';
import React, { useEffect, useRef, useState } from 'react';
import { Dimensions, Vibration, Platform, StyleSheet, View } from 'react-native';
import Canvas, { CanvasRenderingContext2D } from 'react-native-canvas';

enum CameraType {
  back = 'back',
  front = 'front',
}

export default function App() {
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [type, setType] = useState<CameraType>(CameraType.back);
  const context = useRef<CanvasRenderingContext2D | null>(null);
  const canvas = useRef<Canvas | null>(null);

  useEffect(() => {
    const loadModel = async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      if (status !== 'granted') {
        console.log('Camera permission denied');
        return;
      }
      await tf.ready();
      const loadedModel = await cocoSsd.load();
      setModel(loadedModel);
    };
    loadModel();
  }, []);
  

  function handleCameraStream(images: any) {
    const loop = async () => {
      const nextImageTensor = images.next().value;

      if (!model || !nextImageTensor) throw new Error('no model');

      model
        .detect(nextImageTensor)
        .then((predictions) => {
          drawRectangle(predictions, nextImageTensor);
          for (const prediction of predictions) {
            notifyUser(prediction.class);
          }
        })
        .catch((err) => {
          console.log(err);
        });

      requestAnimationFrame(loop);
    };
    loop();
  }

  function drawRectangle(
    predictions: cocoSsd.DetectedObject[],
    nextImageTensor: any
  ) {
    if (!context.current || !canvas.current) {
      console.log('no context or canvas');
      return;
    }

    const screenWidth = Dimensions.get('window').width;
    const screenHeight = Dimensions.get('window').height;

    const scaleWidth = screenWidth / nextImageTensor.shape[1];
    const scaleHeight = screenHeight / nextImageTensor.shape[0];

    const flipHorizontal = Platform.OS === 'ios' ? false : true;

    context.current.clearRect(0, 0, screenWidth, screenHeight);

    for (const prediction of predictions) {
      const [x, y, w, h] = prediction.bbox;

      const boundingBoxX = flipHorizontal
        ? canvas.current.width - x * scaleWidth - w * scaleWidth
        : x * scaleWidth;
      const boundingBoxY = y * scaleHeight;

      context.current.strokeRect(
        boundingBoxX,
        boundingBoxY,
        w * scaleWidth,
        h * scaleHeight
      );

      context.current.fillText(
        prediction.class,
        boundingBoxX - 5,
        boundingBoxY - 5
      );
    }
  }

  const handleCanvas = async (can: Canvas | null) => {
    if (can) {
      can.width = Dimensions.get('window').width;
      can.height = Dimensions.get('window').height;
      const ctx = can.getContext('2d') as CanvasRenderingContext2D;
      context.current = ctx;
      ctx.strokeStyle = 'red';
      ctx.fillStyle = 'red';
      ctx.lineWidth = 3;
      canvas.current = can;
    }
  };

  return (
    <View style={styles.container}>
      <Camera
        style={styles.camera}
        type={type}
      />
      <Canvas style={styles.canvas} ref={handleCanvas} />
    </View>
  );
}

function notifyUser(objectClass: string) {
  const sound = new Audio('notification_sound.mp3');
  sound.play();

  Vibration.vibrate([500, 500, 500]);
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  camera: {
    width: '100%',
    height: '100%',
  },
  canvas: {
    position: 'absolute',
    zIndex: 1000000,
    width: '100%',
    height: '100%',
  },
});
