#include <TensorFlowLite.h>
#include "model.h"  // Replace with the actual model header file
#include "image_data_zero.h"
#include "image_data_two.h"
#include "image_data_three.h"
#include "image_data_five.h"
#include "image_data_eight.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 50 * 1024;  // Adjust based on your model's requirements
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const char* image_names[] = {"0", "2", "3", "5", "8"};
const uint8_t* images[] = {image_data_zero, image_data_two, image_data_three, image_data_five, image_data_eight};  // Replace with actual image data arrays
const int num_images = 5;
}  // namespace

void setup() {
  Serial.begin(115200);
  tflite::InitializeTarget();

  // Load the model
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Set up the resolver and interpreter
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate tensor buffers
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Get pointers to the model's input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Debug prints
  Serial.print("Input shape: ");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    Serial.print(" ");
  }
  Serial.println();

  Serial.print("Output shape: ");
  for (int i = 0; i < output->dims->size; i++) {
    Serial.print(output->dims->data[i]);
    Serial.print(" ");
  }
  Serial.println();
}

void loadImageToTensor(const uint8_t* image_data) {
  for (int i = 0; i < input->bytes; i++) {
    input->data.uint8[i] = image_data[i];
  }
}

void classifyImage(const char* image_name) {
  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed");
    return;
  }

  // Find the class with the maximum probability
  uint8_t max_index = 0;
  uint8_t max_value = output->data.uint8[0];
  for (int i = 1; i < output->dims->data[1]; i++) {
    if (output->data.uint8[i] > max_value) {
      max_value = output->data.uint8[i];
      max_index = i;
    }
  }

  // Print the results
  Serial.print("Image file name: ");
  Serial.println(image_name);
  Serial.print("Predicted class: ");
  Serial.println(max_index);
  Serial.print("Class probability: ");
  Serial.println(max_value / 255.0);  // Assuming the output is quantized to uint8
}

void loop() {
  for (int i = 0; i < num_images; i++) {
    // Load the image into the tensor
    loadImageToTensor(images[i]);

    // Classify the image
    classifyImage(image_names[i]);

    // Delay to avoid spamming the serial monitor
    delay(1000);
  }
}
