#include <Arduino.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"
#include "image_data_zero.h"
#include "image_data_two.h"
#include "image_data_three.h"
#include "image_data_five.h"
#include "image_data_eight.h"

// Memory area for the TFLite tensor arena
constexpr int kArenaSize = 10 * 1024;
uint8_t tensor_arena[kArenaSize];

// Input Data
const char* image_names[] = {"Img Number 0", "Img Number 2", "Img Number 3", "Img Number 5", "Img Number 8"};
const uint8_t* images[] = {image_data_zero, image_data_two, image_data_three, image_data_five, image_data_eight};
const int num_images = 5;

// TFLite globals
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

void setup() {
  Serial.begin(9600);

  // Load the model
  const tflite::Model* model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model version does not match Schema");
    return;
  }

  // Define the operations resolver
  static tflite::AllOpsResolver resolver;

  // Build the interpreter
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return;
  }

  // Get pointers to the model's input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Debug: print input and output tensor details
  Serial.print("Input tensor dimensions: ");
  for (int i = 0; i < input->dims->size; ++i) {
    Serial.print(input->dims->data[i]);
    if (i < input->dims->size - 1) {
      Serial.print(" x ");
    }
  }
  Serial.println();

  Serial.print("Output tensor dimensions: ");
  for (int i = 0; i < output->dims->size; ++i) {
    Serial.print(output->dims->data[i]);
    if (i < output->dims->size - 1) {
      Serial.print(" x ");
    }
  }
  Serial.println();
}

void loop() {
  for (int i = 0; i < num_images; i++) {

    // Load the image into the tensor
    for (int j = 0; j < input->bytes; j++) {
    input->data.uint8[j] = images[i][j];
    }

    // Run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      Serial.println("Invoke failed");
      return;
    }

    // Get the maximum probability
    int8_t max_index = 0;
    int8_t max_value = output->data.int8[0];
    for (int i = 1; i < output->dims->data[1]; ++i) {
      int8_t value = output->data.int8[i];
      if (value > max_value) {
        max_value = value;
        max_index = i;
      }
    }

    //Convert the probabiltity into float
    float scale = output->params.scale;
    int32_t zero_point = output->params.zero_point;
    float max_value_f = (max_value - zero_point) * scale;

    //Print image file name, class

    Serial.print("Image file name: ");
    Serial.println(image_names[i]);
    Serial.print("Predicted class: ");
    Serial.println(max_index);
    Serial.print("Probability: ");
    Serial.println(max_value_f);

    delay(1000);
  }
}