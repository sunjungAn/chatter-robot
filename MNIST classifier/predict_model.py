from tensorflow.keras.models import load_model

_, (x_test, y_test) = mnist.load_data()
x_test = x_test/255.0

model = load_model('mnist_model.h5')
model.evaluate(x_test, y_test, verbose = 2)

plt.imshow(x_test[20], cmap = 'gray')
plt.show()

picks = [20]
predict = model.predict_classes(x_test[picks])
print("손글씨 이미지 예측값: ", predict)
