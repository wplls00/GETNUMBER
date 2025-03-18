// Получаем элементы страницы
const canvas = document.getElementById('drawing-canvas');
const ctx = canvas.getContext('2d');
const clearButton = document.getElementById('clear-button');
const predictButton = document.getElementById('predict-button');
const resultSpan = document.getElementById('result');
const statusSpan = document.getElementById('status');
let isDrawing = false;

// Очищаем canvas белым цветом
function clearCanvas() {
  ctx.fillStyle = '#FFFFFF'; // Белый фон
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  console.log('Canvas очищен:', ctx.getImageData(0, 0, canvas.width, canvas.height).data.every(pixel => pixel === 255));
}

// Очищаем canvas при загрузке страницы
clearCanvas();

// Кнопка "Очистить"
clearButton.addEventListener('click', () => {
  clearCanvas();
  resultSpan.textContent = '?';
});

// Обработчики событий для рисования
canvas.addEventListener('mousedown', () => isDrawing = true);
canvas.addEventListener('mouseup', () => isDrawing = false);
canvas.addEventListener('mousemove', draw);

function draw(event) {
  if (!isDrawing) return;
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;

  ctx.fillStyle = '#000000'; // Чёрный цвет
  ctx.fillRect(x, y, 20, 20); // Размер кисти
}

// Загружаем модель как Graph Model
let model;
(async function loadModel() {
  try {
    const modelUrl = './web_model/model.json'; // Убедитесь, что путь правильный
    model = await tf.loadGraphModel(modelUrl);
    statusSpan.textContent = 'Модель загружена! Можно рисовать.';
  } catch (error) {
    console.error('Ошибка загрузки модели:', error);
    statusSpan.textContent = 'Ошибка загрузки модели. Проверьте консоль.';
  }
})();

// Предобработка изображения
function preprocessCanvas(canvas) {
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  return tf.tidy(() => {
    let tensor = tf.browser.fromPixels(imageData, 1) // Градации серого (1 канал)
      .resizeNearestNeighbor([28, 28])               // Уменьшаем до 28x28 пикселей
      .toFloat()
      .div(255.0)                                    // Нормализуем значения пикселей
      .expandDims(0);                                // Добавляем размерность batch (форма [1, 28, 28, 1])

    console.log('Форма тензора:', tensor.shape);     // Отладочное сообщение для формы
    console.log('Значения тензора:', tensor.arraySync()); // Отладочное сообщение для значений
    return tensor;
  });
}

// Визуализация входного тензора
function visualizeTensor(tensor) {
  const canvas = document.createElement('canvas');
  canvas.width = 28;
  canvas.height = 28;
  const ctx = canvas.getContext('2d');

  const imageData = ctx.createImageData(28, 28);
  const data = tensor.reshape([28, 28]).arraySync();

  for (let i = 0; i < 28; i++) {
    for (let j = 0; j < 28; j++) {
      const pixelValue = Math.floor(data[i][j] * 255);
      const index = (i * 28 + j) * 4;
      imageData.data[index] = pixelValue;     // R
      imageData.data[index + 1] = pixelValue; // G
      imageData.data[index + 2] = pixelValue; // B
      imageData.data[index + 3] = 255;       // A
    }
  }

  ctx.putImageData(imageData, 0, 0);
  document.body.appendChild(canvas); // Добавляем canvas на страницу
}

// Кнопка "Распознать"
predictButton.addEventListener('click', async () => {
  if (!model) {
    alert('Модель ещё не загружена. Пожалуйста, подождите.');
    return;
  }

  // Предобработка изображения
  const imageTensor = preprocessCanvas(canvas);

  // Визуализация входного тензора
  visualizeTensor(imageTensor);

  // Получаем предсказание
  console.log('Форма тензора:', imageTensor.shape); // Отладочное сообщение
  const prediction = await model.predict(imageTensor).data();
  const result = prediction.indexOf(Math.max(...prediction)); // Находим наиболее вероятную цифру
  resultSpan.textContent = result; // Показываем результат
});
