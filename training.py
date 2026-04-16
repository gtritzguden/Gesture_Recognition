from mediapipe_model_maker import gesture_recognizer

dataset_path = "./dataset/"
print("chargement images")
data = gesture_recognizer.Dataset.from_folder(dirname=dataset_path,
                                              hparams=gesture_recognizer.HandDataPreprocessingParams())
train_data,remaining_data = data.split(0.8)
test_data,validation_data = remaining_data.split(0.5)

print("Début Entrainement")
hparams = gesture_recognizer.HParams(export_dir="my_model",epochs=10)
options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
model = gesture_recognizer.GestureRecognizer.create(train_data=train_data,
                                                    validation_data=validation_data,
                                                    options=options)

model.export_model()
print("Fin entrainement")